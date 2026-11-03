# Лабораторна робота №2 — Docker + Speech Commands

## Що реалізовано
- CNN на PyTorch для 4 класів Google Speech Commands: `yes`, `no`, `up`, `down`.
- Mel-спектрограми 64 mel bins, 16 kHz, 1 секунда.
- Навчання, оцінка, збереження `model.pt`, `labels.json`, `metrics.json`.
- FastAPI: `GET /health`, `GET /model-info`, `POST /predict`, Swagger `/docs`.
- Docker multi-stage та single-stage, healthcheck, непривілейований користувач.
- Benchmark mean/p50/p95 та GitHub Actions.

## Передумови
Встановити Docker Desktop (Linux containers), Python 3.10, Git (для CI).
**Увага:** перша збірка завантажує Google Speech Commands v0.02 (~2.4 ГБ), навчає модель і може тривати довго. Потрібні вільне місце та інтернет. Для демонстрації спочатку використайте `EPOCHS=3`, для фінального експерименту `EPOCHS=10`.

## PowerShell — запуск у папці lab2
```powershell
cd 'C:\Users\User\Desktop\4 курс 1 семестр\проект з ітерацій\lab2'
docker build --build-arg EPOCHS=10 -t kws-api:1.0.0 .
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
```

Перевірка прогнозу: скопіюйте WAV-зразки з контейнера (`docker cp kws:/app/samples/. ./samples/`) або використайте власний WAV зі словом `yes`:
```powershell
curl.exe -X POST -F 'file=@samples/yes.wav' http://localhost:8000/predict
```
Якщо локального `samples/yes.wav` немає, використайте:
```powershell
docker build --target trainer --build-arg EPOCHS=10 -t kws-trainer:1.0.0 .
docker create --name kws-trainer-temp kws-trainer:1.0.0
docker cp kws-trainer-temp:/build/samples/. ./samples/
docker rm kws-trainer-temp
```

## Порівняння
```powershell
docker build -f Dockerfile.single --build-arg EPOCHS=10 -t kws-single:1.0.0 .
docker images
docker history kws-api:1.0.0
docker history kws-single:1.0.0
docker stats --no-stream kws
python -m pip install requests==2.32.3 pytest==8.3.2
python scripts/benchmark.py --output benchmark-multi.json
python -m pytest tests/ -v
```
Для single-stage запустіть окремий контейнер на тому самому порту після зупинки multi-stage, потім повторіть benchmark. Для локального запуску встановіть `requirements-api.txt`, скопіюйте `artifacts` із контейнера (`docker cp kws:/app/artifacts/. ./artifacts/`) та виконайте `python -m uvicorn src.api:app --port 8000`. Порівнюйте ту саму модель та однакові WAV. Для локальної оцінки тестового набору потрібен завантажений датасет.

## Звіт
У `ЗВІТ_шаблон.md` є таблиця для фактичних результатів. Не підставляйте припущені числа. Зробіть скріншоти збірки, статусу healthy, `/docs`, метрик, CI.

## Зауваження
Поріг accuracy в CI — 50% для короткого навчання на 3 епохи; це початкове значення, яке потрібно обґрунтувати або змінити за фактичними результатами. Dockerfile.single зберігає датасет і train-залежності, тому його розмір очікувано більший. Перевірка CI можлива після публікації репозиторію на GitHub.
