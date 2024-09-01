# PBL6 FLASK SERVER

## How to run the application:

### Prerequisites

-   Python 3.12.3 or higher
-   Pip (Python package manager)

### 1. Clone the repository:

```bash
git clone https://github.com/vtvl-pbl6/pbl6-ai.git
```

### 2. Change directory to project:

```bash
cd pbl6-ai
```

### 3. Create .env file:

Create a `.env` file in the root directory of the project with examples in `.env.example` file.

### 4. Run using Docker Compose (recommended for development):

```bash
docker-compose up -d
```

### 5. Run using command line:

#### 5.1. Set up a virtual environment

```bash
python3 -m venv my-env
source my-env/bin/activate
```

#### 5.2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 5.3. Run the Application:

```bash
python3 app.py
```

### 6. Access the API:

-   Once the application is running, the API is available at `http://localhost:8081/api`.
