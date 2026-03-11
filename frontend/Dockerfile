FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy application files (though overriden by bind mount in dev)
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
