"""Social Listener — fetches Twitter/X cashtag mentions and Reddit posts.

Sources:
- Twitter/X: cashtag mentions via Twitter API v2 (bearer token auth)
- Reddit: Google search via SerpAPI with `site:reddit.com` filter, then
  scrapes the actual Reddit post pages for content and engagement data.
  This replaces the PRAW approach since Reddit API is hard to access.

Graceful degradation: if a platform's credentials are missing, that
platform is silently skipped. The other platform's results are still returned.

Output: list[SocialPost] — locked Phase 2 contract.
"""
import logging
import os
import re
from datetime import UTC, datetime

import requests
from bs4 import BeautifulSoup

from app.schemas.social_post import SocialPost

logger = logging.getLogger(__name__)

_TWITTER_MAX_RESULTS = 100
_REDDIT_SUBREDDITS = [
    "IndianStreetBets",     # Most active Indian stock subreddit
    "IndianStockMarket",    # Growing community
    "IndiaInvestments",     # Long-term investing focused
    "DalalStreet",          # Indian market discussions
    "stocks",               # Global stocks — catches Indian tickers too
]
_REDDIT_MAX_RESULTS = 15


def fetch_social(ticker: str, days: int = 21, company_name: str = "") -> list[SocialPost]:
    """Fetch social media posts mentioning a given stock ticker.

    Uses the stock alias generator to discover all common names for the stock,
    then searches across:
    1. Reddit (5 subreddits via SerpAPI or direct scraping)
    2. Twitter/X (via SerpAPI Google search with site:twitter.com filter)
    3. Google web discussions (fallback if Reddit+Twitter yield nothing)

    Args:
        ticker: NSE/BSE ticker (e.g. 'RELIANCE', 'SBIN', 'ETERNAL').
        days: Number of past days to search.
        company_name: Optional override for company name (if empty, auto-resolved).

    Returns:
        Combined list of SocialPost instances from all available platforms.
    """
    # Resolve stock aliases for better search coverage
    from app.scrapers.stock_alias import get_stock_info
    stock_info = get_stock_info(ticker)

    # Use provided company_name or the resolved one
    effective_company_name = company_name or stock_info.company_name

    posts: list[SocialPost] = []

    # 1. Reddit — use all names to search multiple subreddits
    reddit_posts = _fetch_reddit_via_serp(ticker, days, stock_info.all_names)
    posts.extend(reddit_posts)

    # 2. Twitter/X — search Google for tweets mentioning the stock
    twitter_posts = _fetch_twitter_via_google(stock_info, days)
    posts.extend(twitter_posts)

    # 3. If both Reddit and Twitter found nothing, try general Google discussions
    if not reddit_posts and not twitter_posts:
        google_posts = _fetch_google_discussions(ticker, effective_company_name, days)
        posts.extend(google_posts)

    logger.info("fetch_social(%s, days=%d) → %d posts", ticker, days, len(posts))
    return posts


# ── Twitter/X via Google Search ────────────────────────────────────────────────


def _fetch_twitter_via_google(stock_info, days: int) -> list[SocialPost]:
    """Search Google via SerpAPI for Twitter/X posts mentioning the stock.

    Uses site:twitter.com OR site:x.com filter with all known stock names.
    This replaces the Twitter API (paid) with free Google scraping.
    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        logger.info("SERP_API_KEY not set — skipping Twitter/X Google search")
        return []

    # Build search query using all aliases
    names = stock_info.all_names[:4]  # Limit to avoid overly long queries
    name_query = " OR ".join(f'"{n}"' for n in names)
    query = f"({name_query}) (site:twitter.com OR site:x.com)"

    # Time filter
    if days <= 7:
        tbs = "qdr:w"
    elif days <= 30:
        tbs = "qdr:m"
    else:
        tbs = "qdr:m3"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 15,
        "tbs": tbs,
    }

    posts: list[SocialPost] = []
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        organic_results = data.get("organic_results", [])
        logger.info("SerpAPI found %d Twitter/X results for %s", len(organic_results), stock_info.ticker)

        for result in organic_results:
            post = _parse_twitter_google_result(result)
            if post:
                posts.append(post)

    except Exception as exc:
        logger.warning("Twitter/X Google search failed for %s: %s", stock_info.ticker, exc)

    return posts


def _parse_twitter_google_result(result: dict) -> SocialPost | None:
    """Parse a Google result from Twitter/X into a SocialPost."""
    try:
        url = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        if not url:
            return None

        # Extract tweet content: Google often shows "username on X: tweet text"
        content = snippet or title

        # Try to extract author from title ("@username on X" pattern)
        author = ""
        author_match = re.search(r'(@\w+)', title)
        if author_match:
            author = author_match.group(1)
        elif " on X:" in title:
            author = title.split(" on X:")[0].strip()
        elif " on Twitter:" in title:
            author = title.split(" on Twitter:")[0].strip()

        # Try to get date
        date_str = result.get("date", "")
        post_date = _parse_date(date_str) or datetime.now(tz=UTC)

        # Extract status ID from URL
        status_match = re.search(r'/status/(\d+)', url)
        post_id = status_match.group(1) if status_match else re.sub(r'[^a-zA-Z0-9]', '', url[-20:])

        return SocialPost(
            platform="twitter",
            post_id=post_id,
            content=content,
            author=author,
            date=post_date,
            likes=0,
            comments=0,
            url=url,
        )
    except Exception as exc:
        logger.debug("Failed to parse Twitter Google result: %s", exc)
        return None


# ── Reddit via SerpAPI + Scraping ──────────────────────────────────────────────


def _fetch_reddit_via_serp(ticker: str, days: int, search_names: list[str] | None = None) -> list[SocialPost]:
    """Fetch Reddit posts by searching Google via SerpAPI with site:reddit.com filter.

    Uses all known stock names (aliases) for broader search coverage.
    Falls back to scraping Reddit search directly if SerpAPI key is not set.
    """
    serp_api_key = os.getenv("SERP_API_KEY")
    names = search_names or [ticker]

    if serp_api_key:
        return _reddit_via_serpapi(names, days, serp_api_key)
    else:
        logger.info("SERP_API_KEY not set — trying direct Reddit search scraping")
        return _reddit_via_direct_scrape(ticker, days)


def _reddit_via_serpapi(search_names: list[str], days: int, api_key: str) -> list[SocialPost]:
    """Use SerpAPI to Google-search Reddit for stock mentions using all aliases."""
    posts: list[SocialPost] = []

    # Map days to Google's tbs time filter parameter
    if days <= 7:
        tbs = "qdr:w"   # past week
    elif days <= 30:
        tbs = "qdr:m"   # past month
    else:
        tbs = "qdr:m3"  # past 3 months

    # Build OR query from all stock names
    name_query = " OR ".join(f'"{n}"' for n in search_names[:5])  # Limit to 5

    for subreddit in _REDDIT_SUBREDDITS:
        query = f"({name_query}) site:reddit.com/r/{subreddit}"
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": _REDDIT_MAX_RESULTS,
            "tbs": tbs,
        }

        try:
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            organic_results = data.get("organic_results", [])
            logger.info(
                "SerpAPI returned %d results for '%s' in r/%s",
                len(organic_results), search_names[0], subreddit,
            )

            for result in organic_results:
                post = _parse_serp_reddit_result(result, subreddit)
                if post:
                    posts.append(post)

        except Exception as exc:
            logger.warning("SerpAPI Reddit search failed for r/%s: %s", subreddit, exc)

    # Optionally enrich with scraped engagement data
    posts = _enrich_with_scraping(posts)

    return posts


def _parse_serp_reddit_result(result: dict, subreddit: str) -> SocialPost | None:
    """Parse a single SerpAPI Google organic result into a SocialPost."""
    try:
        url = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        # Skip non-post URLs (subreddit pages, wiki, etc.)
        if not re.search(r"reddit\.com/r/\w+/comments/", url):
            return None

        # Extract post ID from URL: /r/subreddit/comments/POST_ID/...
        match = re.search(r"/comments/(\w+)", url)
        post_id = match.group(1) if match else url

        # SerpAPI sometimes includes the date in the snippet
        date_str = result.get("date", "")
        post_date = _parse_date(date_str) or datetime.now(tz=UTC)

        # Content = title + snippet from Google
        content = title
        if snippet:
            content = f"{title}\n{snippet}"

        return SocialPost(
            platform="reddit",
            post_id=post_id,
            content=content,
            author="",  # Enriched by scraping if possible
            date=post_date,
            likes=0,     # Enriched by scraping
            comments=0,  # Enriched by scraping
            url=url,
        )
    except Exception as exc:
        logger.debug("Failed to parse SerpAPI result: %s — %s", result, exc)
        return None


def _enrich_with_scraping(posts: list[SocialPost]) -> list[SocialPost]:
    """Scrape old.reddit.com for each post to get upvotes, comments, author.

    Uses old.reddit.com because it renders server-side HTML (new reddit is
    JS-heavy and hard to scrape). Fails gracefully — posts without
    enrichment keep their SerpAPI data.
    """
    enriched = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    }

    for post in posts:
        try:
            # Convert URL to old.reddit.com
            old_url = post.url.replace("www.reddit.com", "old.reddit.com")
            if "old.reddit.com" not in old_url:
                old_url = old_url.replace("reddit.com", "old.reddit.com")

            resp = requests.get(old_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                enriched.append(post)
                continue

            soup = BeautifulSoup(resp.text, "lxml")

            # Extract upvotes from the score element
            score_el = soup.select_one(".score.unvoted")
            score = 0
            if score_el:
                score_text = score_el.get("title", "0")
                try:
                    score = int(score_text)
                except (ValueError, TypeError):
                    pass

            # Extract number of comments
            comments_el = soup.select_one("a.comments")
            num_comments = 0
            if comments_el:
                comments_text = comments_el.get_text(strip=True)
                match = re.search(r"(\d+)", comments_text)
                if match:
                    num_comments = int(match.group(1))

            # Extract author
            author_el = soup.select_one("a.author")
            author = author_el.get_text(strip=True) if author_el else "[deleted]"

            # Extract self-text body if available
            selftext_el = soup.select_one(".usertext-body .md")
            body = selftext_el.get_text(strip=True) if selftext_el else ""
            content = post.content
            if body:
                content = f"{post.content}\n\n{body[:500]}"  # Cap at 500 chars

            enriched.append(SocialPost(
                platform="reddit",
                post_id=post.post_id,
                content=content,
                author=author,
                date=post.date,
                likes=score,
                comments=num_comments,
                url=post.url,
            ))
            logger.debug("Enriched Reddit post %s: score=%d comments=%d", post.post_id, score, num_comments)

        except Exception as exc:
            logger.debug("Failed to enrich post %s: %s", post.post_id, exc)
            enriched.append(post)  # Keep original data on failure

    return enriched


def _reddit_via_direct_scrape(ticker: str, days: int) -> list[SocialPost]:
    """Fallback: scrape Reddit search results directly without SerpAPI.

    Uses old.reddit.com/r/{subreddit}/search?q={ticker}&sort=new&t=month
    which renders server-side and doesn't require JavaScript.
    """
    posts: list[SocialPost] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    }

    # Map days to Reddit's time filter
    if days <= 7:
        time_filter = "week"
    elif days <= 30:
        time_filter = "month"
    else:
        time_filter = "year"

    for subreddit in _REDDIT_SUBREDDITS:
        search_url = (
            f"https://old.reddit.com/r/{subreddit}/search"
            f"?q={ticker}&sort=new&restrict_sr=on&t={time_filter}"
        )

        try:
            resp = requests.get(search_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning("Reddit search returned %d for r/%s", resp.status_code, subreddit)
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            search_results = soup.select("div.search-result-link")[:_REDDIT_MAX_RESULTS]

            for result in search_results:
                post = _parse_reddit_search_result(result, subreddit)
                if post:
                    posts.append(post)

            logger.info(
                "Direct Reddit search found %d posts for '%s' in r/%s",
                len(search_results), ticker, subreddit,
            )

        except Exception as exc:
            logger.warning("Reddit direct scrape failed for r/%s: %s", subreddit, exc)

    return posts


def _parse_reddit_search_result(result_el, subreddit: str) -> SocialPost | None:
    """Parse a single old.reddit.com search result element."""
    try:
        # Title and URL
        title_el = result_el.select_one("a.search-title")
        if not title_el:
            return None

        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        if url and url.startswith("/"):
            url = f"https://old.reddit.com{url}"

        # Post ID from URL
        match = re.search(r"/comments/(\w+)", url)
        post_id = match.group(1) if match else ""

        # Metadata
        score_el = result_el.select_one("span.search-score")
        score_text = score_el.get_text(strip=True) if score_el else "0"
        score_match = re.search(r"(\d+)", score_text)
        score = int(score_match.group(1)) if score_match else 0

        comments_el = result_el.select_one("a.search-comments")
        comments_text = comments_el.get_text(strip=True) if comments_el else "0"
        comments_match = re.search(r"(\d+)", comments_text)
        num_comments = int(comments_match.group(1)) if comments_match else 0

        author_el = result_el.select_one("a.author")
        author = author_el.get_text(strip=True) if author_el else "[deleted]"

        # Date
        time_el = result_el.select_one("time")
        date_str = time_el.get("datetime", "") if time_el else ""
        post_date = _parse_date(date_str) or datetime.now(tz=UTC)

        return SocialPost(
            platform="reddit",
            post_id=post_id or url,
            content=title,
            author=author,
            date=post_date,
            likes=score,
            comments=num_comments,
            url=url.replace("old.reddit.com", "www.reddit.com"),
        )
    except Exception as exc:
        logger.debug("Failed to parse Reddit search result: %s", exc)
        return None


def _parse_date(date_str: str) -> datetime | None:
    """Best-effort date parsing from various source formats."""
    if not date_str:
        return None

    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%b %d, %Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue

    return None


# ── Google Search Fallback ─────────────────────────────────────────────────────


def _fetch_google_discussions(ticker: str, company_name: str, days: int) -> list[SocialPost]:
    """Fallback: Search Google via SerpAPI for stock discussions when Reddit yields nothing.

    Searches for "{ticker} OR {company_name} stock discussion" and returns
    results as SocialPost objects with platform="web". This captures
    blog posts, forum threads, social media posts, and articles that
    mention the stock.

    Requires SERP_API_KEY. Returns empty list if not set.
    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        logger.info("SERP_API_KEY not set — skipping Google discussion search")
        return []

    search_term = f'"{ticker}"'
    if company_name:
        search_term = f'"{ticker}" OR "{company_name}"'

    query = f"{search_term} stock discussion India"

    # Map days to Google's tbs time filter
    if days <= 7:
        tbs = "qdr:w"
    elif days <= 30:
        tbs = "qdr:m"
    else:
        tbs = "qdr:m3"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 15,
        "tbs": tbs,
    }

    posts: list[SocialPost] = []
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        organic_results = data.get("organic_results", [])
        logger.info(
            "Google search returned %d results for '%s' discussions",
            len(organic_results), ticker,
        )

        for result in organic_results:
            post = _parse_google_result(result)
            if post:
                posts.append(post)

    except Exception as exc:
        logger.warning("Google discussion search failed for %s: %s", ticker, exc)

    return posts


def _parse_google_result(result: dict) -> SocialPost | None:
    """Parse a single SerpAPI Google organic result into a SocialPost."""
    try:
        url = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        if not url or not title:
            return None

        # Determine source platform from URL
        if "reddit.com" in url:
            platform = "reddit"
        elif "twitter.com" in url or "x.com" in url:
            platform = "twitter"
        else:
            platform = "web"

        # Extract post ID from URL
        post_id = re.sub(r'[^a-zA-Z0-9]', '', url[-20:])

        date_str = result.get("date", "")
        post_date = _parse_date(date_str) or datetime.now(tz=UTC)

        content = title
        if snippet:
            content = f"{title}\n{snippet}"

        # Source domain as author
        source = result.get("source", "") or result.get("displayed_link", "")

        return SocialPost(
            platform=platform,
            post_id=post_id,
            content=content,
            author=source,
            date=post_date,
            likes=0,
            comments=0,
            url=url,
        )
    except Exception as exc:
        logger.debug("Failed to parse Google result: %s — %s", result, exc)
        return None

