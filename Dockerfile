# Dockerfile for Dedalus Labs MCP deployment
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port (though MCP uses stdio, not HTTP)
EXPOSE 8000

# Run the MCP server
CMD ["python", "main.py"]
