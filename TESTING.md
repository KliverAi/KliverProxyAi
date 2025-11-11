# Testing Guide

## Resumen de Tests

✅ **20/20 tests pasando (100%)**
📊 **Cobertura: 54%**

## Estructura de Tests

```
tests/
├── __init__.py         # Inicialización
├── conftest.py         # Fixtures compartidas
├── test_health.py      # Tests de endpoints de salud (2 tests)
├── test_chat.py        # Tests de chat endpoint (11 tests)
├── test_cache.py       # Tests de cache creation (7 tests)
└── README.md           # Documentación detallada
```

## Comandos Rápidos

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar todos los tests
make test

# Tests con reporte de cobertura
make test-cov

# Ver coverage en el navegador
make coverage

# Tests en modo watch (re-ejecuta al cambiar archivos)
make test-watch

# Solo tests que fallaron en la última ejecución
make test-failed

# Tests con salida verbose
make test-verbose
```

## Tests Implementados

### Health Endpoints (2 tests)
- ✅ Root endpoint retorna información correcta
- ✅ Health check retorna status

### Chat Endpoint (11 tests)
- ✅ Validación de campos requeridos
- ✅ Validación de roles inválidos
- ✅ Auto-detección de provider (OpenAI, Gemini, Claude)
- ✅ Override explícito de provider
- ✅ Estructura de request válida
- ✅ Temperatura personalizada
- ✅ Context cache parameter
- ✅ Modelo TokenAiServiceUsageInfo
- ✅ Valores por defecto de TokenAiServiceUsageInfo

### Cache Creation (7 tests)
- ✅ Validación de campos requeridos
- ✅ Rechazo de modelos no-Gemini
- ✅ Validación de contenido vacío
- ✅ Formato de contenido de texto
- ✅ Upload de archivos de texto
- ✅ Rechazo de archivos binarios
- ✅ Formato JSON array

## Fixtures Disponibles

Las siguientes fixtures están disponibles en `conftest.py`:

- `client`: TestClient de FastAPI
- `mock_api_key`: API key de prueba
- `sample_chat_request`: Request de chat OpenAI
- `sample_gemini_request`: Request de chat Gemini
- `sample_text_file`: Archivo de texto temporal
- `sample_pdf_file`: Archivo PDF temporal

## Escribir Nuevos Tests

### Ejemplo básico

```python
def test_my_feature(client):
    """Test description"""
    response = client.get("/my-endpoint")
    assert response.status_code == 200
```

### Ejemplo con fixture

```python
def test_with_api_key(client, mock_api_key):
    """Test with API key"""
    data = {"api_key": mock_api_key, "model": "gpt-4"}
    response = client.post("/api/endpoint", json=data)
    assert response.status_code in [200, 400]
```

## Configuración de Coverage

El objetivo es mantener >80% de cobertura. Actualmente: **54%**

Para mejorar la cobertura, se necesitan tests de integración con APIs reales (o mocks más complejos).

## CI/CD

Para integrar con GitHub Actions, agrega este archivo:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install Poetry
        run: pip install poetry
      - name: Install dependencies
        run: make install
      - name: Run tests
        run: make test-cov
```

## Notas

- Los tests están simplificados para no requerir APIs reales
- Tests de integración completos requerirían API keys válidas
- La cobertura puede mejorarse agregando tests de integración
