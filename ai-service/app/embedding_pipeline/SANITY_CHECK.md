# Embedding sanity check

Fecha de ejecución: 2026-11-24. Modelo: `text-embedding-3-small`, con la
dimensión por defecto de 1536. Resultados obtenidos ejecutando
`scripts/compare.py` para cada pareja; la similitud coseno se calcula con la
biblioteca estándar de Python.

| Pareja | Texto 1 | Texto 2 | Similitud coseno | Expectativa orientativa |
| --- | --- | --- | ---: | --- |
| A — cercanos | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | **0.5957** | Alta, aproximadamente > 0.6 |
| B — no relacionados | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | **0.1920** | Baja, aproximadamente < 0.4 |
| C — genéricos | Backend services | API development | **0.5407** | Sin expectativa fija |

La pareja A queda ligeramente por debajo del umbral orientativo, pese a su
cercanía semántica. La pareja B obtiene una similitud baja, como se esperaba.

La pareja C alcanza un valor relativamente alto y cercano al de A. Al ser
frases cortas y genéricas, contienen poca información que distinga los
contextos; esto puede dificultar la recuperación precisa. Los umbrales dependen
del modelo, de los textos y del caso de uso, y no deben tratarse como reglas
universales.
