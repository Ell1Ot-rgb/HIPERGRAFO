#!/bin/bash
# Script de Verificación rápida del puente Hipergrafo-Colab

echo "╔════════════════════════════════════════════════════╗"
echo "║  VERIFICADOR DE PUENTE HIPERGRAFO ↔️  COLAB       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

echo "📋 Estado del Proyecto:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "src/neural/ColabBridge.ts" ]; then
    echo "✅ ColabBridge.ts existe"
else
    echo "❌ ColabBridge.ts NO ENCONTRADO"
fi

if [ -f "src/neural/IntegradorHipergrafoColo.ts" ]; then
    echo "✅ IntegradorHipergrafoColo.ts existe"
else
    echo "❌ IntegradorHipergrafoColo.ts NO ENCONTRADO"
fi

if [ -f "src/neural/configColab.ts" ]; then
    echo "✅ configColab.ts existe"
else
    echo "❌ configColab.ts NO ENCONTRADO"
fi

if [ -f "src/pruebas/prueba_colab.ts" ]; then
    echo "✅ prueba_colab.ts existe"
else
    echo "❌ prueba_colab.ts NO ENCONTRADO"
fi

if [ -f "PUENTE_COLAB.md" ]; then
    echo "✅ PUENTE_COLAB.md (Documentación) existe"
else
    echo "❌ PUENTE_COLAB.md NO ENCONTRADO"
fi

echo ""
echo "🔧 Compilación:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

npm run build > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Compilación exitosa"
else
    echo "❌ Errores de compilación"
    npm run build | grep error
fi

echo ""
echo "📡 Próximos Pasos:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  En Google Colab:"
echo "   → Abre: https://colab.research.google.com"
echo "   → Pega el código de PUENTE_COLAB.md"
echo "   → Copia la URL de ngrok"
echo ""
echo "2️⃣  En Codespaces:"
echo "   → Actualiza src/neural/configColab.ts con la URL"
echo "   → Ejecuta: npx ts-node src/pruebas/prueba_colab.ts"
echo ""
echo "3️⃣  Verifica la conexión:"
echo "   → Deberías ver: ✅ Puente con Colab ACTIVO"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  ¡Listo para la comunicación IA ↔️ IA!             ║"
echo "╚════════════════════════════════════════════════════╝"
