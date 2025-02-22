# Use an official Python runtime as a parent image
FROM python:3.10

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start FastAPI when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
