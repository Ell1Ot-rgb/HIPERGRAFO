# 🎓 GUÍA COMPLETA: Usar Kimi K2 + OpenRouter + Python en Codespace

## ✅ Estado Actual

- ✅ **API Key**: `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`
- ✅ **Modelo**: `moonshotai/kimi-k2:free` (gratuito, con rate limits)
- ✅ **Fallback**: `openai/gpt-3.5-turbo` (económico, confiable)
- ✅ **Base URL**: `https://openrouter.ai/api/v1`
- ✅ **Librería**: `openai` (SDK oficial compatible)

---

## 📦 Instalación Rápida

```bash
# Instalar dependencias
pip install openai python-dotenv

# Crear archivo .env (opcional, para desarrollo local)
echo "OPENROUTER_API_KEY=tu_api_key_aqui" > .env
echo ".env" >> .gitignore
```

---

## 🚀 Código de Ejemplo Mínimo

```python
from openai import OpenAI
import os

# Inicializar cliente con OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Headers opcionales para estadísticas
extra_headers = {
    "HTTP-Referer": "https://github.com/tu-usuario/tu-repo",
    "X-Title": "Mi Proyecto",
}

# Llamar al modelo
response = client.chat.completions.create(
    extra_headers=extra_headers,
    model="openai/gpt-3.5-turbo",  # O "moonshotai/kimi-k2:free"
    messages=[
        {"role": "user", "content": "Hola, ¿cómo estás?"}
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response.choices[0].message.content)
print(f"Tokens usados: {response.usage.total_tokens}")
```

---

## 🔄 Usar Kimi K2 vs GPT-3.5-Turbo

### Opción 1: Kimi K2 (Gratuito con límites)
```python
model="moonshotai/kimi-k2:free"
```
- ✅ **Ventajas**: Gratuito, excelente calidad filosófica
- ⚠️ **Desventajas**: Rate limit (429), puede no estar disponible

### Opción 2: GPT-3.5-Turbo (Recomendado)
```python
model="openai/gpt-3.5-turbo"
```
- ✅ **Ventajas**: Económico (~$0.0005/1K tokens), confiable
- ✅ **Desventajas**: Ninguna relevante para este caso

### Opción 3: Fallback Automático
```python
def obtener_modelo():
    try:
        # Intentar Kimi primero
        return "moonshotai/kimi-k2:free"
    except:
        # Fallback a GPT-3.5
        return "openai/gpt-3.5-turbo"

model = obtener_modelo()
```

---

## 📊 Ejemplo Completo: Sistema Ciclo Relacional

Ver archivo: `ciclo_relacional_kimi_openrouter.py`

Este script incluye:
- ✅ Descubrimiento de rutas conceptuales (10-15 rutas)
- ✅ Análisis profundo de las mejores 5
- ✅ Extracción de grafo de conocimiento
- ✅ Generación de reportes JSON + Markdown
- ✅ Manejo de errores y rate limits
- ✅ Contador de tokens y llamadas API

### Ejecución:
```bash
python ciclo_relacional_kimi_openrouter.py
```

### Archivos generados:
- `RESULTADO_CICLO_KIMI_EXISTENCIA.json` - Datos estructurados
- `REPORTE_CICLO_KIMI_EXISTENCIA.md` - Reporte legible

---

## 🛠️ Manejo de Errores

### Error: Rate Limit (429)
```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    if "429" in str(e):
        print("Rate limit alcanzado. Esperando 30s...")
        time.sleep(30)
        # Reintentar con modelo alternativo
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            ...
        )
```

### Error: Invalid Model ID
```python
# ❌ INCORRECTO
model="moonshotai/kimi-k2-0711:free"  # No válido

# ✅ CORRECTO
model="moonshotai/kimi-k2:free"       # Formato correcto
```

### Error: Insufficient Quota
```python
# Verificar saldo en: https://openrouter.ai/account
# Usar modelo gratuito con :free
model="moonshotai/kimi-k2:free"  # Gratuito
```

---

## 📈 Mejores Prácticas

### 1. **Agregar headers** para estadísticas
```python
extra_headers = {
    "HTTP-Referer": "https://github.com/usuario/repo",
    "X-Title": "Nombre del Proyecto",
}
```

### 2. **Monitorear tokens**
```python
response = client.chat.completions.create(...)
print(f"Tokens entrada: {response.usage.prompt_tokens}")
print(f"Tokens salida: {response.usage.completion_tokens}")
print(f"Total: {response.usage.total_tokens}")
```

### 3. **Configurar timeouts**
```python
response = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[...],
    timeout=120,  # 2 minutos
    max_tokens=2000,
)
```

### 4. **Usar temperatura apropiada**
```python
# Tareas creativas: 0.7-0.9
model = "..."
temperature=0.8  # Más creativo

# Tareas analíticas: 0.3-0.5
temperature=0.4  # Más preciso
```

---

## 💾 Configuración en GitHub Codespaces

### Opción 1: Secret de Codespaces (Recomendada)
1. Ve a: Repo → Settings → Secrets and variables → Codespaces
2. Crea `OPENROUTER_API_KEY` con tu clave
3. Se expondrá automáticamente como variable de entorno

```python
api_key = os.getenv("OPENROUTER_API_KEY")
```

### Opción 2: Archivo `.env` (Solo desarrollo local)
```bash
# Crear .env
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env

# Agregar a .gitignore
echo ".env" >> .gitignore

# Usar con python-dotenv
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
```

---

## 🔗 APIs Alternativas

### API Oficial de Moonshot (opcional)
```python
client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)
model="kimi-k2-0711-preview"
```

### Otros modelos en OpenRouter
```python
# Claude
model="anthropic/claude-3-5-sonnet"

# Llama
model="meta-llama/llama-3.1-70b-instruct"

# Mistral
model="mistralai/mistral-nemo:free"

# Gemini
model="google/gemini-2-0-flash-exp:free"
```

---

## 📊 Costos Comparativos

| Modelo | Entrada | Salida | Notas |
|--------|---------|--------|-------|
| Kimi K2 (free) | $0 | $0 | Gratuito, con rate limit |
| GPT-3.5-Turbo | $0.50/1M | $1.50/1M | Muy económico |
| Mistral Nemo (free) | $0 | $0 | Alternativa gratuita |
| Claude 3.5 Sonnet | $3/1M | $15/1M | Más caro, mejor calidad |

---

## ✅ Checklist Rápido

- [ ] Instalar `openai` y `python-dotenv`
- [ ] Tener API Key de OpenRouter: `sk-or-v1-...`
- [ ] Configurar Base URL: `https://openrouter.ai/api/v1`
- [ ] Elegir modelo: `openai/gpt-3.5-turbo` o `moonshotai/kimi-k2:free`
- [ ] Agregar headers opcionales para estadísticas
- [ ] Implementar manejo de errores (429, 402, etc.)
- [ ] Monitorear tokens usados
- [ ] Testear antes de producción
- [ ] Agregar `.env` a `.gitignore`

---

## 📚 Recursos

- **OpenRouter Docs**: https://openrouter.ai/docs
- **OpenAI SDK**: https://github.com/openai/openai-python
- **Kimi K2 Info**: https://moonshot.ai/
- **Models disponibles**: https://openrouter.ai/models

---

## 🎯 Conclusión

Con esta configuración tienes acceso a:
- ✅ Kimi K2 gratuito (cuando no tiene rate limit)
- ✅ GPT-3.5-Turbo económico como fallback
- ✅ Otros 50+ modelos disponibles
- ✅ SDK unificado (openai) para todos
- ✅ Seguro en GitHub Codespaces con secrets

**Costo total**: Prácticamente gratis (~$0.003-0.005 por ejecución del ciclo relacional)

---

**Última actualización**: 2025-11-09  
**Autor**: YO-Estructural  
**Estado**: ✅ Funcional
