# Kliver.AI Chat API

API REST construida con FastAPI y LangChain para interactuar con modelos de OpenAI, Google Gemini y Anthropic Claude.

## Características

- Endpoint POST `/api/chat` para enviar mensajes al modelo de IA
- **Soporte multi-proveedor**: OpenAI, Google Gemini y Anthropic Claude
- **Detección automática** del proveedor basado en el nombre del modelo
- **Logging completo**: Monitoreo de requests, tiempos de respuesta y errores
- Clases pythonic equivalentes a ChatMessage de C#
- Integración con LangChain para manejo de mensajes
- CORS habilitado para llamadas desde C#
- Validación de datos con Pydantic

## Requisitos

- Python >= 3.12
- Poetry

## Instalación

Las dependencias ya están instaladas con Poetry. Si necesitas reinstalarlas:

```bash
poetry install
```

## Uso

### Iniciar el servidor

```bash
poetry run python run.py
```

El servidor estará disponible en `http://localhost:8000`

### Documentación interactiva

Una vez iniciado el servidor, puedes acceder a:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoint

### POST /api/chat

Envía una lista de mensajes al modelo de IA y obtiene una respuesta.

#### Ejemplo con OpenAI (GPT-4):

**Request Body:**

```json
{
  "model": "gpt-4",
  "api_key": "sk-...",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7
}
```

#### Ejemplo con Google Gemini:

**Request Body:**

```json
{
  "model": "gemini-pro",
  "api_key": "AIza...",
  "messages": [
    {
      "role": "user",
      "content": "Hola, ¿cómo estás?"
    }
  ],
  "temperature": 0.9
}
```

**Nota:** El proveedor se detecta automáticamente según el nombre del modelo. También puedes especificarlo explícitamente:

```json
{
  "model": "gemini-pro",
  "api_key": "AIza...",
  "provider": "gemini",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 1.0
}
```

#### Ejemplo con Anthropic Claude:

**Request Body:**

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "api_key": "sk-ant-...",
  "messages": [
    {
      "role": "user",
      "content": "Hola, ¿cómo estás?"
    }
  ],
  "temperature": 1.0
}
```

#### Ejemplo con GPT-5 (Modelo de Razonamiento):

**Request Body:**

```json
{
  "model": "gpt-5",
  "api_key": "sk-proj-...",
  "messages": [
    {
      "role": "user",
      "content": "Resuelve este problema: Si tengo 3 manzanas y compro el doble, ¿cuántas tengo?"
    }
  ]
}
```

**Nota:** Los modelos GPT-5 y o-series usan configuración especial automática:
- `reasoning_effort="minimal"` (hardcodeado)
- `verbosity="low"` (hardcodeado, solo GPT-5)
- `temperature=1.0` (forzado, ignora el valor del request)

Ver [examples_gpt5_reasoning.md](examples_gpt5_reasoning.md) para más ejemplos.

**Response (para todos los proveedores):**

```json
{
  "content": "I'm doing well, thank you! How can I help you today?",
  "model": "gpt-4",
  "role": "assistant"
}
```

## Roles disponibles

- `system`: Mensajes del sistema
- `user`: Mensajes del usuario
- `assistant`: Respuestas del asistente
- `tool`: Mensajes de herramientas
- `developer`: Mensajes del desarrollador

## Parámetro Temperature

El parámetro `temperature` controla la aleatoriedad de las respuestas del modelo:

- **0.0 - 0.3**: Respuestas muy deterministas y enfocadas. Ideal para tareas que requieren precisión (código, análisis, datos)
- **0.4 - 0.7**: Balance entre creatividad y coherencia (valor por defecto: 0.7)
- **0.8 - 1.2**: Respuestas más creativas y variadas. Bueno para escritura creativa, brainstorming
- **1.3 - 2.0**: Máxima aleatoriedad y creatividad. Puede producir respuestas menos coherentes

**Ejemplo:**
```json
{
  "temperature": 0.2  // Para respuestas precisas y consistentes
}
```

## Ejemplo de llamada desde C#

```csharp
using System.Net.Http.Json;

public class ChatRole
{
    public const string System = "system";
    public const string User = "user";
    public const string Assistant = "assistant";
    public const string Tool = "tool";
    public const string Developer = "developer";
}

public record ChatMessage(string Role, string Content);

public record ChatRequest(
    string Model,
    string ApiKey,
    List<ChatMessage> Messages,
    string? Provider = null,  // Optional: "openai" or "gemini"
    double Temperature = 0.7  // Optional: 0.0 to 2.0
);

public record ChatResponse(
    string Content,
    string Model,
    string Role
);

// Uso con OpenAI
var httpClient = new HttpClient { BaseAddress = new Uri("http://localhost:8000") };

var request = new ChatRequest(
    Model: "gpt-4.1",
    ApiKey: "sk-...",
    Messages: new List<ChatMessage>
    {
        new ChatMessage("system", "You are a helpful assistant."),
        new ChatMessage("user", "Hello!")
    }
);

var response = await httpClient.PostAsJsonAsync("/api/chat", request);
var chatResponse = await response.Content.ReadFromJsonAsync<ChatResponse>();

Console.WriteLine(chatResponse.Content);

// Uso con Google Gemini
var geminiRequest = new ChatRequest(
    Model: "gemini-pro",
    ApiKey: "AIza...",
    Messages: new List<ChatMessage>
    {
        new ChatMessage("user", "Hola!")
    }
);

var geminiResponse = await httpClient.PostAsJsonAsync("/api/chat", geminiRequest);
var geminiChatResponse = await geminiResponse.Content.ReadFromJsonAsync<ChatResponse>();

Console.WriteLine(geminiChatResponse.Content);

// Uso con Anthropic Claude
var claudeRequest = new ChatRequest(
    Model: "claude-3-7-sonnet-20250219",
    ApiKey: "sk-ant-...",
    Messages: new List<ChatMessage>
    {
        new ChatMessage("user", "Hola!")
    },
    Temperature: 1.0
);

var claudeResponse = await httpClient.PostAsJsonAsync("/api/chat", claudeRequest);
var claudeChatResponse = await claudeResponse.Content.ReadFromJsonAsync<ChatResponse>();

Console.WriteLine(claudeChatResponse.Content);
```

## Modelos disponibles

### OpenAI - Modelos de Razonamiento (GPT-5 y o-series)
- `gpt-5` (último modelo con razonamiento avanzado)
- `gpt-5-mini` (balance entre costo y capacidad)
- `gpt-5-nano` (más rápido y económico)
- `o1` (modelo de razonamiento profundo)
- `o1-mini` (razonamiento rápido)
- `o3`, `o4-mini` (próximas generaciones)

**Nota:** Los modelos de razonamiento usan configuración especial hardcodeada:
- `reasoning_effort="minimal"` (razonamiento mínimo para máxima velocidad)
- `verbosity="low"` (respuestas cortas y concisas - solo GPT-5)
- `temperature=1.0` (forzado automáticamente)

### OpenAI - Modelos Estándar
- `gpt-4.1` (lanzado en 2025)
- `gpt-4.1-mini` (más económico)
- `gpt-4.1-nano` (el más rápido y barato)
- `gpt-4`
- `gpt-4-turbo-preview`
- `gpt-3.5-turbo`

### Google Gemini
- `gemini-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-2.0-flash-exp`

### Anthropic Claude
- `claude-3-7-sonnet-20250219` (nuevo, con modo de pensamiento extendido)
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

**Nota:** El proveedor se detecta automáticamente. Los modelos que comienzan con "gemini" usan Google AI, "gpt" usan OpenAI, y "claude" usan Anthropic.

## Logging

La API incluye logging completo para monitorear todas las operaciones:

- **Información de cada request**: Modelo, temperatura, número de mensajes
- **Proveedor detectado**: OpenAI, Gemini o Claude
- **Métricas de performance**: Tiempo de respuesta del LLM y tiempo total del request
- **Errores detallados**: Stack traces completos para debugging

Ejemplo de logs (modelo estándar):
```
2025-10-20 19:30:15 - kliver.ai - INFO - New chat request received
2025-10-20 19:30:15 - kliver.ai - INFO - Model: gpt-4.1
2025-10-20 19:30:15 - kliver.ai - INFO - Provider detected: openai
2025-10-20 19:30:15 - kliver.ai - INFO - Calling openai LLM...
2025-10-20 19:30:18 - kliver.ai - INFO - LLM response received in 2.834s
2025-10-20 19:30:18 - kliver.ai - INFO - Total request duration: 2.856s
```

Ejemplo de logs (modelo de razonamiento GPT-5/o1):
```
2025-10-20 20:15:30 - kliver.ai - INFO - Model: gpt-5
2025-10-20 20:15:30 - kliver.ai - INFO - Provider detected: openai
2025-10-20 20:15:30 - kliver.ai - INFO - Detected reasoning model: gpt-5 - Using minimal reasoning_effort and low verbosity
2025-10-20 20:15:30 - kliver.ai - INFO - GPT-5 model - Added verbosity='low'
2025-10-20 20:15:30 - kliver.ai - INFO - Calling openai LLM...
2025-10-20 20:15:33 - kliver.ai - INFO - LLM response received in 2.543s
```

## Notas

- El API key se envía en cada request por seguridad
- CORS está habilitado para todas las origins (configura apropiadamente en producción)
- El parámetro `temperature` es opcional (valor por defecto: 0.7)
- El parámetro `provider` se detecta automáticamente según el modelo, pero puede especificarse explícitamente
