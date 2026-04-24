# MAEKEO LAB — 분쇄도 분석 서버

OpenCV 기반 커피 원두 분쇄도 정밀 분석 서버입니다.

## 로컬 테스트

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

서버 실행 후 http://localhost:8000 접속

## Railway 배포

1. https://railway.app 접속 → GitHub 로그인
2. New Project → Deploy from GitHub repo
3. 이 폴더를 GitHub에 올린 후 연결
4. 자동 배포 완료 → URL 발급

## API 사용법

POST /analyze
Body: { "image": "base64_jpeg_string" }

Response:
{
  "result": {
    "levelKor": "중간",
    "particleSize": 412,
    "uniformity": 78,
    "mokaFit": "최적",
    "advice": "...",
    ...
  }
}
