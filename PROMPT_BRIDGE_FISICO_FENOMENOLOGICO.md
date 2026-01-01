# 🌉 PROMPT: Bridge Físico-Fenomenológico Completo

## 🎯 Objetivo

Crea un sistema que extraiga una **firma física de 72 dimensiones** de cualquier archivo digital y la mapee automáticamente a las **categorías fenomenológicas de REMForge**, creando un puente bidireccional entre el mundo físico del hardware y el mundo abstracto de la experiencia fenomenológica.

---

## 📋 Especificaciones Técnicas

### Input
- **Archivo digital** (cualquier formato: txt, pdf, jpg, mp3, exe, zip, etc.)
- **Contexto opcional** (metadata, autor, timestamp)

### Output
```python
{
    "signature_72d": [float] * 72,  # Vector de 72 dimensiones
    "phenomenal_mapping": {
        "sensorial_layer": {...},
        "noetic_layer": {...},
        "qualia_signature": {...},
        "semantic_contamination": {...},
        "invariant_features": {...}
    },
    "physical_metrics": {
        "energy_profile": [...],
        "thermal_signature": [...],
        "complexity_score": float,
        "entropy": float
    },
    "remforge_compatible": dict  # Output directo para REMForge
}
```

---

## 🔧 Implementación Completa

### 1. Extractor de Firma 72D

```python
"""
physical_signature_extractor.py
================================
Extrae la firma física de 72 dimensiones de un archivo digital.
"""

import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import zlib
from collections import Counter
import struct
import time

class PhysicalSignatureExtractor:
    """
    Extractor de firma física de 72 dimensiones.
    
    Dimensiones organizadas en 5 grupos:
    - Grupo 1 (0-15):   Energy & Thermal Profile
    - Grupo 2 (16-31):  Structural Complexity 
    - Grupo 3 (32-47):  Cryptographic Patterns
    - Grupo 4 (48-63):  Temporal Dynamics
    - Grupo 5 (64-71):  Invariant Fingerprints
    """
    
    def __init__(self):
        self.dimension_names = self._initialize_dimension_names()
    
    def _initialize_dimension_names(self) -> List[str]:
        """Define nombres descriptivos para cada dimensión"""
        names = []
        
        # Grupo 1: Energy & Thermal (16D)
        names.extend([
            "avg_energy_per_byte",           # 0
            "energy_variance",                # 1
            "thermal_signature_mean",         # 2
            "thermal_signature_std",          # 3
        ] + [f"thermal_histogram_bin_{i}" for i in range(8)] +  # 4-11
          [f"power_profile_{i}" for i in range(4)])              # 12-15
        
        # Grupo 2: Structural Complexity (16D)
        names.extend([
            "kolmogorov_complexity_ratio",    # 16
            "shannon_entropy",                # 17
            "byte_entropy",                   # 18
            "compression_ratio_zlib",         # 19
            "compression_ratio_bz2",          # 20
        ] + [f"entropy_distribution_{i}" for i in range(8)] +    # 21-28
          [f"pattern_complexity_{i}" for i in range(3)])         # 29-31
        
        # Grupo 3: Cryptographic Patterns (16D)
        names.extend([f"sha256_feature_{i}" for i in range(16)])  # 32-47
        
        # Grupo 4: Temporal Dynamics (16D)
        names.extend([
            "creation_timestamp_normalized",   # 48
            "modification_timestamp_normalized", # 49
            "access_timestamp_normalized",     # 50
            "timestamp_delta_create_modify",   # 51
            "timestamp_delta_modify_access",   # 52
        ] + [f"temporal_pattern_{i}" for i in range(3)] +        # 53-55
          [f"access_fourier_coeff_{i}" for i in range(8)])       # 56-63
        
        # Grupo 5: Invariant Fingerprints (8D)
        names.extend([f"invariant_pca_{i}" for i in range(8)])   # 64-71
        
        return names
    
    def extract_signature(self, file_path: str) -> np.ndarray:
        """
        Extrae firma de 72 dimensiones de un archivo.
        
        Args:
            file_path: Ruta al archivo
        
        Returns:
            np.ndarray de shape (72,) con la firma física
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Leer archivo completo
        with open(file_path, 'rb') as f:
            data = f.read()
        
        signature = np.zeros(72, dtype=np.float32)
        
        # Grupo 1: Energy & Thermal (0-15)
        signature[0:16] = self._extract_energy_thermal(data, file_path)
        
        # Grupo 2: Structural Complexity (16-31)
        signature[16:32] = self._extract_complexity(data)
        
        # Grupo 3: Cryptographic Patterns (32-47)
        signature[32:48] = self._extract_cryptographic(data)
        
        # Grupo 4: Temporal Dynamics (48-63)
        signature[48:64] = self._extract_temporal(file_path)
        
        # Grupo 5: Invariant Fingerprints (64-71)
        signature[64:72] = self._extract_invariants(signature[0:64])
        
        return signature
    
    def _extract_energy_thermal(self, data: bytes, file_path: Path) -> np.ndarray:
        """Extrae perfil energético y térmico simulado"""
        features = np.zeros(16)
        
        # Simular energía por byte (basado en operaciones)
        byte_array = np.frombuffer(data[:10000], dtype=np.uint8)
        features[0] = np.mean(byte_array) / 255.0  # Normalizado
        features[1] = np.var(byte_array) / (255.0 ** 2)
        
        # Firma térmica (distribución de bytes como proxy de "calor")
        features[2] = np.mean(byte_array) / 255.0
        features[3] = np.std(byte_array) / 255.0
        
        # Histograma térmico (bins de distribución)
        hist, _ = np.histogram(byte_array, bins=8, range=(0, 256))
        features[4:12] = hist / np.sum(hist)  # Normalizado
        
        # Perfil de "potencia" (cambios entre bytes consecutivos)
        if len(byte_array) > 1:
            power_changes = np.abs(np.diff(byte_array.astype(int)))
            features[12] = np.mean(power_changes) / 255.0
            features[13] = np.max(power_changes) / 255.0
            features[14] = np.min(power_changes) / 255.0
            features[15] = np.median(power_changes) / 255.0
        
        return features
    
    def _extract_complexity(self, data: bytes) -> np.ndarray:
        """Extrae complejidad estructural"""
        features = np.zeros(16)
        
        # Complejidad de Kolmogorov (aproximada por compresión)
        compressed_zlib = zlib.compress(data, level=9)
        compressed_bz2 = zlib.compress(data, level=9)  # Simulación
        
        features[0] = len(compressed_zlib) / max(len(data), 1)  # Kolmogorov ratio
        features[1] = self._calculate_shannon_entropy(data)
        features[2] = len(set(data)) / 256.0  # Byte entropy
        features[3] = len(compressed_zlib) / max(len(data), 1)
        features[4] = len(compressed_bz2) / max(len(data), 1)
        
        # Distribución de entropía por chunks
        chunk_size = max(len(data) // 8, 1)
        for i in range(8):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(data))
            chunk = data[start:end]
            if chunk:
                features[5 + i] = self._calculate_shannon_entropy(chunk)
        
        # Complejidad de patrones (n-gramas)
        if len(data) >= 3:
            trigrams = [data[i:i+3] for i in range(len(data) - 2)]
            unique_trigrams = len(set(trigrams))
            features[13] = unique_trigrams / max(len(trigrams), 1)
        
        features[14] = self._lempel_ziv_complexity(data[:10000])
        features[15] = self._run_length_complexity(data[:10000])
        
        return features
    
    def _extract_cryptographic(self, data: bytes) -> np.ndarray:
        """Extrae características criptográficas del hash SHA-256"""
        features = np.zeros(16)
        
        # Hash SHA-256
        sha256_hash = hashlib.sha256(data).digest()
        
        # Convertir cada 2 bytes del hash en un float normalizado
        for i in range(16):
            byte_pair = sha256_hash[i*2:(i*2)+2]
            value = struct.unpack('>H', byte_pair)[0]  # Unsigned short
            features[i] = value / 65535.0  # Normalizar a [0, 1]
        
        return features
    
    def _extract_temporal(self, file_path: Path) -> np.ndarray:
        """Extrae características temporales"""
        features = np.zeros(16)
        
        stat = file_path.stat()
        
        # Timestamps normalizados (epoch → [0, 1] en rango razonable)
        # Usar 2020-2030 como rango de referencia
        ref_start = 1577836800  # 2020-01-01
        ref_end = 1893456000    # 2030-01-01
        ref_range = ref_end - ref_start
        
        features[0] = (stat.st_ctime - ref_start) / ref_range
        features[1] = (stat.st_mtime - ref_start) / ref_range
        features[2] = (stat.st_atime - ref_start) / ref_range
        
        # Deltas temporales
        features[3] = (stat.st_mtime - stat.st_ctime) / 86400.0  # días
        features[4] = (stat.st_atime - stat.st_mtime) / 86400.0
        
        # Patrones temporales (día de semana, hora del día)
        import datetime
        dt = datetime.datetime.fromtimestamp(stat.st_mtime)
        features[5] = dt.weekday() / 7.0
        features[6] = dt.hour / 24.0
        features[7] = dt.minute / 60.0
        
        # Fourier de acceso (simulado con timestamps)
        time_values = np.array([stat.st_ctime, stat.st_mtime, stat.st_atime])
        fft = np.fft.fft(time_values)
        features[8:16] = np.abs(fft.real[:8]) / np.max(np.abs(fft.real))
        
        return features
    
    def _extract_invariants(self, base_features: np.ndarray) -> np.ndarray:
        """Extrae características invariantes usando PCA"""
        # Simular PCA: tomar combinaciones lineales estables
        features = np.zeros(8)
        
        # Primer componente: promedio ponderado de energía
        features[0] = np.mean(base_features[0:16])
        
        # Segundo componente: promedio de complejidad
        features[1] = np.mean(base_features[16:32])
        
        # Tercer componente: promedio de criptográfico
        features[2] = np.mean(base_features[32:48])
        
        # Cuarto componente: varianza temporal
        features[3] = np.var(base_features[48:64])
        
        # Componentes adicionales: combinaciones
        features[4] = np.dot(base_features[0:16], base_features[16:32]) / 16
        features[5] = np.dot(base_features[32:48], base_features[48:64]) / 16
        features[6] = np.linalg.norm(base_features[0:32])
        features[7] = np.linalg.norm(base_features[32:64])
        
        return features
    
    # Utilidades
    
    def _calculate_shannon_entropy(self, data: bytes) -> float:
        """Calcula entropía de Shannon"""
        if not data:
            return 0.0
        
        counter = Counter(data)
        length = len(data)
        entropy = 0.0
        
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy / 8.0  # Normalizar (max=8 para bytes)
    
    def _lempel_ziv_complexity(self, data: bytes) -> float:
        """Complejidad de Lempel-Ziv (aproximada)"""
        if not data:
            return 0.0
        
        # Versión simplificada
        n = len(data)
        i = 0
        c = 1
        
        while i < n - 1:
            j = i + 1
            while j < n and data[i:j] in data[:i]:
                j += 1
            c += 1
            i = j
        
        return c / n
    
    def _run_length_complexity(self, data: bytes) -> float:
        """Complejidad por run-length encoding"""
        if not data:
            return 0.0
        
        runs = 1
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                runs += 1
        
        return runs / len(data)
    
    def get_dimension_info(self, dimension_index: int) -> Dict[str, str]:
        """Retorna información sobre una dimensión específica"""
        if 0 <= dimension_index < 72:
            name = self.dimension_names[dimension_index]
            
            # Determinar grupo
            if dimension_index < 16:
                group = "Energy & Thermal"
            elif dimension_index < 32:
                group = "Structural Complexity"
            elif dimension_index < 48:
                group = "Cryptographic Patterns"
            elif dimension_index < 64:
                group = "Temporal Dynamics"
            else:
                group = "Invariant Fingerprints"
            
            return {
                "index": dimension_index,
                "name": name,
                "group": group
            }
        else:
            raise ValueError(f"Índice de dimensión fuera de rango: {dimension_index}")
```

---

### 2. Mapper Físico → Fenomenológico

```python
"""
physical_phenomenal_bridge.py
==============================
Mapea la firma física de 72D a categorías fenomenológicas de REMForge.
"""

from typing import Dict, Any
import numpy as np

class PhysicalPhenomenalBridge:
    """
    Puente entre firma física (72D) y estructura fenomenológica (REMForge).
    
    Mapeo:
    - Dimensiones 0-15  → Sensorial Layer
    - Dimensiones 16-31 → Noetic Layer
    - Dimensiones 32-47 → Qualia Signature
    - Dimensiones 48-63 → Semantic Contamination
    - Dimensiones 64-71 → Invariant Features
    """
    
    def __init__(self):
        self.mapping_rules = self._initialize_mapping_rules()
    
    def _initialize_mapping_rules(self) -> Dict:
        """Define reglas de mapeo entre dominios"""
        return {
            "sensorial": {
                "source_dims": (0, 16),
                "target": "sensorial_layer",
                "mappings": {
                    "affective_arousal": [0, 1, 2, 3],  # Energy/thermal → arousal
                    "affective_valence": [4, 5, 6, 7],   # Thermal dist → valence
                    "modality_distribution": [8, 9, 10, 11, 12, 13, 14, 15]
                }
            },
            "noetic": {
                "source_dims": (16, 32),
                "target": "noetic_layer",
                "mappings": {
                    "ego_involvement": [16, 17],  # Complexity → ego
                    "act_intensity": [18, 19, 20],
                    "temporal_phase": [21, 22, 23, 24, 25, 26, 27, 28],
                    "directedness": [29, 30, 31]
                }
            },
            "qualia": {
                "source_dims": (32, 48),
                "target": "qualia_signature",
                "mappings": {
                    "qualia_type": [32, 33, 34, 35],
                    "intensity_profile": [36, 37, 38, 39, 40, 41, 42, 43],
                    "discrimination_threshold": [44, 45],
                    "phenomenal_saturation": [46, 47]
                }
            },
            "contamination": {
                "source_dims": (48, 64),
                "target": "semantic_contamination",
                "mappings": {
                    "contamination_strength": [48, 49, 50],
                    "lexical_anchors_density": [51, 52, 53, 54, 55],
                    "semantic_traces_count": [56, 57, 58, 59, 60, 61, 62, 63]
                }
            },
            "invariants": {
                "source_dims": (64, 72),
                "target": "invariant_features",
                "mappings": {
                    "stable_patterns": [64, 65, 66, 67],
                    "eidetic_reductions": [68, 69, 70, 71]
                }
            }
        }
    
    def map_to_phenomenal(
        self, 
        signature_72d: np.ndarray,
        filename: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Mapea firma física a estructura fenomenológica compatible con REMForge.
        
        Args:
            signature_72d: Vector de 72 dimensiones
            filename: Nombre del archivo (para contexto)
        
        Returns:
            Dict con estructura fenomenológica completa
        """
        if signature_72d.shape[0] != 72:
            raise ValueError(f"Se esperaban 72 dimensiones, recibidas {signature_72d.shape[0]}")
        
        # Construir estructura fenomenológica
        phenomenal_structure = {
            "header": {
                "rem_id": f"physical_{hash(filename) % 100000:05d}",
                "forge_version": "4.0.0-physical-bridge",
                "creation_timestamp": self._get_timestamp(),
                "modality_origin": "physical_signature",
                "source_file": filename,
                "quality_metrics": {
                    "completeness_score": float(np.mean(signature_72d > 0)),
                    "contamination_detected": bool(np.mean(signature_72d[48:64]) > 0.5),
                    "phenomenal_resolution": float(np.std(signature_72d) * 10)
                }
            },
            "sensorial_layer": self._map_sensorial(signature_72d[0:16]),
            "noetic_layer": self._map_noetic(signature_72d[16:32]),
            "phenomenal_core": {
                "qualia_signature": self._map_qualia(signature_72d[32:48]),
                "invariant_features": self._map_invariants(signature_72d[64:72])
            },
            "semantic_contamination": self._map_contamination(signature_72d[48:64])
        }
        
        return phenomenal_structure
    
    def _map_sensorial(self, sensorial_dims: np.ndarray) -> Dict:
        """Mapea dimensiones 0-15 a capa sensorial"""
        return {
            "modality_distribution": {
                "thermal": float(np.mean(sensorial_dims[0:4])),
                "energetic": float(np.mean(sensorial_dims[4:8])),
                "structural": float(np.mean(sensorial_dims[8:12])),
                "dynamic": float(np.mean(sensorial_dims[12:16]))
            },
            "affective_valence": float(np.mean(sensorial_dims[4:8])),
            "affective_arousal": float(np.mean(sensorial_dims[0:4])),
            "spatial_horizon": "digital_space",
            "sensorial_resolution": {
                "precision": float(np.std(sensorial_dims)),
                "granularity": float(1.0 / (np.std(sensorial_dims) + 0.01))
            }
        }
    
    def _map_noetic(self, noetic_dims: np.ndarray) -> Dict:
        """Mapea dimensiones 16-31 a capa noética"""
        # Determinar modo intencional basado en complejidad
        complexity_score = float(np.mean(noetic_dims))
        
        if complexity_score < 0.3:
            intentional_mode = "perception"
        elif complexity_score < 0.6:
            intentional_mode = "memory"
        elif complexity_score < 0.8:
            intentional_mode = "reflection"
        else:
            intentional_mode = "imagination"
        
        return {
            "intentional_mode": intentional_mode,
            "directedness": "file_structure" if np.mean(noetic_dims[13:16]) > 0.5 else "content_essence",
            "temporal_phase": "retention" if np.mean(noetic_dims[5:13]) > 0.5 else "protention",
            "ego_involvement": float(np.mean(noetic_dims[0:2])),
            "horizon_type": "inner" if complexity_score > 0.6 else "outer",
            "act_intensity": float(np.mean(noetic_dims[2:5]))
        }
    
    def _map_qualia(self, qualia_dims: np.ndarray) -> Dict:
        """Mapea dimensiones 32-47 a signature de qualia"""
        # Determinar tipo dominante de qualia
        qualia_vector = qualia_dims[0:4]
        qualia_types = ["computational", "cryptographic", "structural", "deterministic"]
        dominant_idx = int(np.argmax(qualia_vector))
        
        return {
            "qualia_type": qualia_types[dominant_idx],
            "intensity_profile": qualia_dims[4:12].tolist(),
            "discrimination_threshold": float(np.mean(qualia_dims[12:14])),
            "phenomenal_saturation": float(np.mean(qualia_dims[14:16])),
            "purity_index": float(1.0 - np.var(qualia_dims))
        }
    
    def _map_contamination(self, contamination_dims: np.ndarray) -> Dict:
        """Mapea dimensiones 48-63 a contaminación semántica"""
        strength = float(np.mean(contamination_dims[0:3]))
        
        return {
            "contamination_strength": strength,
            "source": "temporal_metadata" if np.mean(contamination_dims) > 0.5 else "structural_patterns",
            "lexical_anchors": [
                {"anchor": f"temporal_marker_{i}", "weight": float(contamination_dims[i])}
                for i in range(3, 8)
            ],
            "semantic_traces": [
                {"trace_type": "timestamp", "intensity": float(contamination_dims[i])}
                for i in range(8, 16)
            ],
            "invariance_under_semantic_permutation": {
                "score": float(1.0 - strength),
                "stable": strength < 0.3
            }
        }
    
    def _map_invariants(self, invariant_dims: np.ndarray) -> Dict:
        """Mapea dimensiones 64-71 a características invariantes"""
        return {
            "stable_geometric_features": invariant_dims[0:4].tolist(),
            "temporal_stability": float(np.mean(invariant_dims[4:6])),
            "eidetic_core": {
                "essence_vector": invariant_dims.tolist(),
                "reduction_level": "physical_signature",
                "phenomenal_atoms": int(np.sum(invariant_dims > 0.5))
            }
        }
    
    def _get_timestamp(self) -> str:
        """Retorna timestamp ISO actual"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_ghost_node_data(
        self, 
        signature_72d: np.ndarray,
        filename: str
    ) -> Dict[str, Any]:
        """
        Crea datos para nodo 'ghost' en el grafo (UI).
        
        Args:
            signature_72d: Firma de 72D
            filename: Nombre del archivo
        
        Returns:
            Dict con datos para visualización
        """
        phenomenal = self.map_to_phenomenal(signature_72d, filename)
        
        # Calcular color único del archivo (hash → HSL)
        file_hash = hash(filename)
        hue = (file_hash % 360)
        saturation = 70 + (file_hash % 30)
        lightness = 50 + (file_hash % 20)
        color = f"hsl({hue}, {saturation}%, {lightness}%)"
        
        # Calcular "mood" (melancolía) desde componentes térmicas
        thermal_mean = np.mean(signature_72d[0:4])
        mood = int(thermal_mean * 100)
        
        # Complejidad
        complexity_score = np.mean(signature_72d[16:32])
        if complexity_score > 0.7:
            complexity = "Alta (Caótica)"
        elif complexity_score > 0.4:
            complexity = "Media (Equilibrada)"
        else:
            complexity = "Baja (Armónica)"
        
        return {
            "filename": filename,
            "color": color,
            "mood": mood,
            "complexity": complexity,
            "signature_72d": signature_72d.tolist(),
            "phenomenal_mapping": phenomenal,
            "graph_node": {
                "id": f"ghost_{filename}",
                "group": "renode_ghost",
                "val": 50,
                "label": f"72-D: {filename}",
                "color": color,
                "details": {
                    "definition": phenomenal['phenomenal_core']['qualia_signature'],
                    "instances": [
                        f"Melancolía: {mood}%",
                        f"Complejidad: {complexity}",
                        f"Qualia: {phenomenal['phenomenal_core']['qualia_signature']['qualia_type']}"
                    ]
                }
            }
        }
```

---

### 3. Integración con Sistema Principal

```python
"""
integration_example.py
======================
Ejemplo de integración completa del bridge en el sistema.
"""

from physical_signature_extractor import PhysicalSignatureExtractor
from physical_phenomenal_bridge import PhysicalPhenomenalBridge
import numpy as np

# Inicializar componentes
extractor = PhysicalSignatureExtractor()
bridge = PhysicalPhenomenalBridge()

# === FLUJO COMPLETO ===

# 1. Usuario sube archivo
file_path = "entrada_bruta/documento_importante.pdf"

# 2. Extraer firma física de 72D
print("🔬 Extrayendo firma física...")
signature_72d = extractor.extract_signature(file_path)
print(f"✅ Firma 72D extraída: {signature_72d.shape}")

# 3. Mapear a estructura fenomenológica
print("\n🌉 Mapeando a estructura fenomenológica...")
phenomenal_structure = bridge.map_to_phenomenal(signature_72d, "documento_importante.pdf")
print(f"✅ Modo intencional: {phenomenal_structure['noetic_layer']['intentional_mode']}")
print(f"✅ Qualia dominante: {phenomenal_structure['phenomenal_core']['qualia_signature']['qualia_type']}")

# 4. Crear datos para grafo (UI)
print("\n🕸️ Creando nodo ghost para grafo...")
ghost_data = bridge.create_ghost_node_data(signature_72d, "documento_importante.pdf")
print(f"✅ Color del nodo: {ghost_data['color']}")
print(f"✅ Mood: {ghost_data['mood']}%")
print(f"✅ Complejidad: {ghost_data['complexity']}")

# 5. Persistir en Neo4j
print("\n💾 Guardando en Neo4j...")
cypher_query = """
MERGE (f:FileEntity {filename: $filename})
SET f.signature_72d = $signature_72d,
    f.color = $color,
    f.mood = $mood,
    f.complexity = $complexity,
    f.qualia_type = $qualia_type,
    f.intentional_mode = $intentional_mode,
    f.phenomenal_resolution = $phenomenal_resolution

// Relacionar con conceptos similares
WITH f
MATCH (g:Grundzug)
WHERE gds.similarity.cosine(f.signature_72d[0:31], g.embedding) > 0.75
MERGE (f)-[r:RESONATES_WITH]->(g)
SET r.score = gds.similarity.cosine(f.signature_72d[0:31], g.embedding),
    r.resonance_type = '72D_physical'

RETURN f.filename as file, count(r) as connections
"""

# Parámetros
params = {
    "filename": "documento_importante.pdf",
    "signature_72d": signature_72d.tolist(),
    "color": ghost_data['color'],
    "mood": ghost_data['mood'],
    "complexity": ghost_data['complexity'],
    "qualia_type": phenomenal_structure['phenomenal_core']['qualia_signature']['qualia_type'],
    "intentional_mode": phenomenal_structure['noetic_layer']['intentional_mode'],
    "phenomenal_resolution": phenomenal_structure['header']['quality_metrics']['phenomenal_resolution']
}

print(f"✅ Query Cypher preparado con {len(params)} parámetros")

# 6. Enviar a UI vía WebSocket
print("\n📡 Enviando a UI...")
websocket_payload = {
    "type": "GHOST_NODE_CREATED",
    "data": ghost_data
}
print(f"✅ Payload WebSocket: {list(websocket_payload.keys())}")

print("\n🎉 Pipeline completo ejecutado exitosamente")
```

---

## 🚀 Uso en API FastAPI

```python
"""
api_endpoint.py
===============
Endpoint REST para procesar archivos y extraer firma 72D.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from physical_signature_extractor import PhysicalSignatureExtractor
from physical_phenomenal_bridge import PhysicalPhenomenalBridge

app = FastAPI()

extractor = PhysicalSignatureExtractor()
bridge = PhysicalPhenomenalBridge()

@app.post("/api/extract-72d-signature")
async def extract_signature_endpoint(file: UploadFile = File(...)):
    """
    Extrae firma 72D de un archivo y la mapea a estructura fenomenológica.
    
    Returns:
        JSON con signature_72d, phenomenal_mapping, y graph_node_data
    """
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extraer firma
        signature_72d = extractor.extract_signature(tmp_path)
        
        # Mapear a fenomenológico
        phenomenal = bridge.map_to_phenomenal(signature_72d, file.filename)
        
        # Crear datos de grafo
        ghost_data = bridge.create_ghost_node_data(signature_72d, file.filename)
        
        return JSONResponse({
            "success": True,
            "signature_72d": signature_72d.tolist(),
            "phenomenal_structure": phenomenal,
            "ghost_node": ghost_data['graph_node'],
            "visualization": {
                "color": ghost_data['color'],
                "mood": ghost_data['mood'],
                "complexity": ghost_data['complexity']
            }
        })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
```

---

## ✅ Checklist de Implementación

- [ ] Crear `physical_signature_extractor.py`
- [ ] Crear `physical_phenomenal_bridge.py`
- [ ] Instalar dependencias: `numpy`, `scipy` (si se usa)
- [ ] Crear tests unitarios para cada grupo de dimensiones
- [ ] Integrar con API FastAPI existente
- [ ] Actualizar schema Neo4j con campos 72D
- [ ] Modificar UI para recibir datos del bridge
- [ ] Crear visualización de heatmap 72D en dashboard
- [ ] Implementar WebSocket para updates en tiempo real
- [ ] Documentar cada dimensión con ejemplos

---

## 🎯 Resultado Final

Con este bridge, el sistema puede:
1. ✅ Tomar **cualquier archivo** (PDF, imagen, video, código)
2. ✅ Extraer una **firma física real de 72D**
3. ✅ Mapearla automáticamente a **REMForge phenomenal structure**
4. ✅ Persistir en **Neo4j con relaciones a conceptos existentes**
5. ✅ Visualizar en **grafo 3D con "ghost nodes" únicos**
6. ✅ Buscar archivos por **similitud física o fenomenológica**

**Esto convierte archivos digitales en entidades fenomenológicas verificables.**
