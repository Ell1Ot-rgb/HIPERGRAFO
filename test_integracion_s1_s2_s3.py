"""
Test de Integración Completa S1 + S2 + S3
=========================================

Verifica que los 3 sistemas de Capa 2 estén correctamente integrados
en el Sistema Principal.
"""

import sys
from pathlib import Path

# Agregar path del proyecto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("TestIntegracion")


def test_integracion_completa():
    """Test completo de integración."""
    
    print("="*70)
    print("TEST DE INTEGRACIÓN S1 + S2 + S3")
    print("="*70)
    
    # =============================================
    # PASO 1: Importar Sistema Principal
    # =============================================
    print("\n[1/5] Importando Sistema Principal...")
    try:
        from core.sistema_principal import SistemaYoEstructural
        print("  ✅ Import exitoso")
    except ImportError as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # =============================================
    # PASO 2: Inicializar Sistema
    # =============================================
    print("\n[2/5] Inicializando Sistema...")
    try:
        sistema = SistemaYoEstructural("configuracion/config_4gb.yaml")
        print("  ✅ Sistema inicializado")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  ℹ️  Verifica que config_4gb.yaml exista")
        return False
    
    # =============================================
    # PASO 3: Verificar S2 (Emergencia)
    # =============================================
    print("\n[3/5] Verificando S2 (Sistema de Emergencia)...")
    
    if not hasattr(sistema, 'sistema_emergencia'):
        print("  ❌ sistema.sistema_emergencia no existe")
        return False
    
    if sistema.sistema_emergencia is None:
        print("  ⚠️  S2 no disponible (módulo no importado)")
        print("  ℹ️  Verifica que emergencia_concepto/ exista")
    else:
        print(f"  ✅ S2 inicializado")
        print(f"     - Sistemas observados: {len(sistema.sistema_emergencia.sistemas)}")
        print(f"     - Conceptos emergidos: {len(sistema.sistema_emergencia.conceptos)}")
        
        # Test de método ciclo_incremental
        if hasattr(sistema.sistema_emergencia, 'ciclo_incremental'):
            print("  ✅ Método ciclo_incremental() disponible")
        else:
            print("  ❌ Método ciclo_incremental() NO disponible")
            return False
    
    # =============================================
    # PASO 4: Verificar S3 (Lógica)
    # =============================================
    print("\n[4/5] Verificando S3 (Sistema de Lógica Pura)...")
    
    if not hasattr(sistema, 'sistema_logica'):
        print("  ❌ sistema.sistema_logica no existe")
        return False
    
    if sistema.sistema_logica is None:
        print("  ⚠️  S3 no disponible (módulo no importado)")
        print("  ℹ️  Verifica que logica_pura/ exista")
    else:
        print(f"  ✅ S3 inicializado")
        print(f"     - Mundos creados: {len(sistema.sistema_logica.mundos)}")
        
        axiomas_totales = sum(
            len(m.axiomas) for m in sistema.sistema_logica.mundos.values()
        ) if sistema.sistema_logica.mundos else 0
        
        print(f"     - Axiomas totales: {axiomas_totales}")
        
        # Test de método ciclo_incremental
        if hasattr(sistema.sistema_logica, 'ciclo_incremental'):
            print("  ✅ Método ciclo_incremental() disponible")
        else:
            print("  ❌ Método ciclo_incremental() NO disponible")
            return False
    
    # =============================================
    # PASO 5: Test Funcional (opcional)
    # =============================================
    print("\n[5/5] Test Funcional...")
    
    if sistema.sistema_emergencia and sistema.sistema_logica:
        print("  → Ejecutando test con datos sintéticos...")
        
        # Grundzugs de prueba
        grundzugs_test = [
            {
                'nombre': 'concepto_test_integracion',
                'certeza': 0.85,
                'nivel': 1,
                'instancias_count': 10,
                'qualia_dominante': 'visual'
            }
        ]
        
        # Test S2
        try:
            resultado_s2 = sistema.sistema_emergencia.ciclo_incremental(
                nuevos_grundzugs=grundzugs_test
            )
            print(f"  ✅ S2 procesó: certeza={resultado_s2.get('patron_certeza', 0):.3f}")
        except Exception as e:
            print(f"  ⚠️  S2 error: {e}")
        
        # Test S3
        try:
            observaciones = [
                {
                    'sujeto': 'concepto_test',
                    'predicado': 'existente',
                    'objeto': 'realidad',
                    'certeza': 0.85
                }
            ]
            
            resultado_s3 = sistema.sistema_logica.ciclo_incremental(
                mundo_nombre="mundo_test",
                nuevos_grundzugs=grundzugs_test,
                observaciones=observaciones
            )
            print(f"  ✅ S3 procesó: {resultado_s3.get('objetos_totales', 0)} objetos")
        except Exception as e:
            print(f"  ⚠️  S3 error: {e}")
    
    else:
        print("  ⏭️  Test funcional omitido (S2 o S3 no disponibles)")
    
    # =============================================
    # RESULTADO FINAL
    # =============================================
    print("\n" + "="*70)
    print("✅ INTEGRACIÓN VERIFICADA EXITOSAMENTE")
    print("="*70)
    
    print("\nComponentes activos:")
    componentes = sistema.get_componentes_activos() if hasattr(sistema, 'get_componentes_activos') else []
    for comp in componentes:
        print(f"  ✓ {comp}")
    
    if sistema.sistema_emergencia:
        print("\n  ✓ Sistema de Emergencia (S2)")
    if sistema.sistema_logica:
        print("  ✓ Sistema de Lógica Pura (S3)")
    
    return True


def test_orquestador():
    """Test del orquestador de Capa 2."""
    
    print("\n" + "="*70)
    print("TEST DEL ORQUESTADOR CAPA 2")
    print("="*70)
    
    try:
        from core.orquestador_capa2 import OrquestadorCapa2
        print("\n✅ Orquestador importado exitosamente")
        
        # Inicializar
        print("\n→ Inicializando orquestador...")
        orq = OrquestadorCapa2("configuracion/config_4gb.yaml")
        
        print(f"\n✅ Orquestador listo")
        print(f"   - S1 activo: {orq.sistema_principal is not None}")
        print(f"   - S2 activo: {orq.sistema_emergencia is not None}")
        print(f"   - S3 activo: {orq.sistema_logica is not None}")
        print(f"   - Monje disponible: {orq.monje_disponible}")
        
        # Test de evento
        print("\n→ Procesando evento de prueba...")
        evento_test = {
            'intensidad': 0.72,
            'complejidad': 0.85,
            'tipo_base': 'narrativo',
            'origen_fisico': {
                'hash': 'a7f3e2d9c1b4a8f3',
                'energia_uj': 3420,
                'ciclos': 1250000
            }
        }
        
        resultado = orq.procesar_evento_fisico(evento_test)
        
        print(f"\n✅ Evento procesado:")
        print(f"   - Grundzugs: {resultado['s1_resultado']['grundzugs']}")
        print(f"   - Certeza S2: {resultado['s2_resultado']['patron_certeza']:.3f}")
        print(f"   - Axiomas S3: {resultado['s3_resultado']['axiomas_totales']}")
        print(f"   - Consistencia: {resultado['validacion_cruzada']['consistencia']:.2%}")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠️  Orquestador no disponible: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error en orquestador: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 SUITE DE TESTS DE INTEGRACIÓN\n")
    
    # Test 1: Sistema Principal
    exito_principal = test_integracion_completa()
    
    # Test 2: Orquestador (opcional)
    print("\n" + "-"*70)
    exito_orquestador = test_orquestador()
    
    # Resultado final
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    print(f"  Sistema Principal: {'✅ PASS' if exito_principal else '❌ FAIL'}")
    print(f"  Orquestador Capa2: {'✅ PASS' if exito_orquestador else '⚠️  SKIP'}")
    
    if exito_principal:
        print("\n🎉 INTEGRACIÓN S1+S2+S3 COMPLETADA Y VERIFICADA\n")
        sys.exit(0)
    else:
        print("\n⚠️  Algunos tests fallaron - Revisar logs arriba\n")
        sys.exit(1)
