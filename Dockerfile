FROM python:3.12-slim

# Install Node for opentk-mcp
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Install Go for CBS MCP server
RUN apt-get install -y golang-go

# Install CBS MCP server
RUN go install github.com/dstotijn/mcp-cbs-cijfers-open-data@latest
ENV PATH="/root/go/bin:${PATH}"

# Pre-install opentk-mcp
RUN npx -y @r-huijts/opentk-mcp --version || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit runs on 8501
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
