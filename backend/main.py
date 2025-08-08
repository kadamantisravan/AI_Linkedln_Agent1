from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import random

# -------------------------------
# 🔑 Load secrets
# -------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ OPENROUTER_API_KEY is missing. Set it in .env or environment variables.")

# -------------------------------
# 📦 Database Setup
# -------------------------------
DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -------------------------------
# 🔒 Password hashing
# -------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str):
    return pwd_context.verify(password, hashed)

# -------------------------------
# 📄 Models
# -------------------------------
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))

class PostDB(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(255))
    industry = Column(String(255))
    topic = Column(String(255))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResumeDB(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    extracted_text = Column(Text)
    analysis = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# 🔑 JWT Helpers
# -------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🛠 Auth Endpoints
# -------------------------------
class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/signup/")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    user = UserDB(username=data.username, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created successfully"}

@app.post("/login/")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.username).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# -------------------------------
# 🚀 LinkedIn Post Generation
# -------------------------------
class PostRequest(BaseModel):
    user_role: str
    industry: str
    topic: str

@app.post("/generate_post/")
def generate_post(data: PostRequest, db: Session = Depends(get_db), user: UserDB = Depends(get_current_user)):
    prompt = f"Create a professional LinkedIn post about {data.topic} in the {data.industry} industry for a person working as {data.user_role}."
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()
    post_content = result['choices'][0]['message']['content']
    new_post = PostDB(role=data.user_role, industry=data.industry, topic=data.topic, content=post_content)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return {"post": post_content, "id": new_post.id}

@app.get("/posts/")
def list_posts(db: Session = Depends(get_db), user: UserDB = Depends(get_current_user)):
    return db.query(PostDB).order_by(PostDB.created_at.desc()).all()

# -------------------------------
# 📄 Resume Upload & Analysis
# -------------------------------
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db), user: UserDB = Depends(get_current_user)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    text = "".join([page.extract_text() or "" for page in reader.pages])

    prompt = f"""
    You are a resume parser. Extract the following details from the resume text:
    - Key Skills
    - Total Years of Experience
    - Highest Education Qualification
    Resume Text:
    {text[:3000]}
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()
    analysis_text = result['choices'][0]['message']['content']
    resume_entry = ResumeDB(filename=file.filename, extracted_text=text, analysis=analysis_text)
    db.add(resume_entry)
    db.commit()
    db.refresh(resume_entry)
    return {"analysis": analysis_text, "id": resume_entry.id}

@app.get("/resumes/")
def list_resumes(db: Session = Depends(get_db), user: UserDB = Depends(get_current_user)):
    return db.query(ResumeDB).order_by(ResumeDB.uploaded_at.desc()).all()

# -------------------------------
# 🧠 Advanced Post by Type
# -------------------------------
class AdvancedPostRequest(BaseModel):
    user_role: str
    industry: str
    topic: str
    post_type: str

@app.post("/generate_advanced_content/")
def generate_advanced_content(data: AdvancedPostRequest, user: UserDB = Depends(get_current_user)):
    prompt = f"""
    You are a LinkedIn content strategist.
    Create a {data.post_type} style LinkedIn post for a {data.user_role} in the {data.industry} industry.
    Topic: {data.topic}
    Make it engaging, professional, and relevant for networking.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.json()
    return {"content": result["choices"][0]["message"]["content"]}

# -------------------------------
# 🌐 Industry Trends
# -------------------------------
class TrendsRequest(BaseModel):
    industry: str

@app.post("/industry_trends/")
def industry_trends(data: TrendsRequest):
    if NEWS_API_KEY:
        url = f"https://newsapi.org/v2/everything?q={data.industry}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        r = requests.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        news_data = r.json()
        trends = [article["title"] for article in news_data.get("articles", [])[:5]]
    else:
        prompt = f"List 5 recent trends in the {data.industry} industry."
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        ai_result = response.json()
        trends = ai_result["choices"][0]["message"]["content"].split("\n")

    return {"trends": [t.strip("-• ") for t in trends if t.strip()]}

# -------------------------------
# 📢 Content Strategy Recommendations
# -------------------------------
class StrategyRequest(BaseModel):
    user_role: str
    industry: str

@app.post("/content_strategy/")
def content_strategy(data: StrategyRequest, user: UserDB = Depends(get_current_user)):
    prompt = f"""
    Suggest a 1-week LinkedIn content plan for a {data.user_role} in the {data.industry} industry.
    Include 5-7 post ideas with short descriptions.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.json()
    return {"strategy": result["choices"][0]["message"]["content"]}

# -------------------------------
# 📈 Mock Post Performance Analytics
# -------------------------------
@app.post("/mock_analytics/")
def mock_analytics(user: UserDB = Depends(get_current_user)):
    return {
        "analytics": {
            "views": random.randint(500, 5000),
            "likes": random.randint(50, 500),
            "comments": random.randint(5, 50)
        }
    }
