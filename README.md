# Лабораторна робота №2 — Docker + Speech Commands

## Опис проєкту

У цій роботі модель машинного навчання для розпізнавання аудіокоманд розгорнуто за допомогою Docker.

Модель **SpeechCNN** створена на PyTorch і розпізнає 4 слова: `yes`, `no`, `up`, `down`. Для навчання використано датасет Google Speech Commands. Аудіо перетворюється на Mel-спектрограму.

Точність навченої моделі — **87,795%**.

## Основні файли

- `src/model.py` — нейронна мережа.
- `src/data.py` — підготовка аудіоданих.
- `src/train.py` — навчання моделі.
- `src/evaluate.py` — оцінювання точності.
- `src/api.py` — API для розпізнавання аудіо.
- `Dockerfile` — multi-stage збірка.
- `Dockerfile.single` — single-stage збірка для порівняння.
- `scripts/benchmark.py` — вимірювання швидкості.
- `.github/workflows/ci.yml` — автоматична перевірка на GitHub.

## Запуск через Docker

```powershell
docker build -t kws-api:1.0.0 .
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps
```

## Перевірка API

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
curl.exe -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
```

Swagger UI: http://localhost:8000/docs

## Порівняння Docker-образів

| Тип збірки | Розмір образу |
|---|---|
| Single-stage | 12,1 ГБ |
| Multi-stage | 1,44 ГБ |

Multi-stage зменшив розмір образу приблизно **у 8,4 раза**, оскільки файли навчання не потрапляють у кінцевий образ.

## Результати

- API успішно запущено в Docker.
- Контейнер отримав статус `healthy`.
- Із 4 тестових аудіофайлів правильно розпізнано 3.
- Проведено порівняння продуктивності локального запуску та Docker.
- GitHub Actions успішно виконав автоматичні перевірки.

## Репозиторій

https://github.com/DianaTsiurpita/lab2-speech-commands
