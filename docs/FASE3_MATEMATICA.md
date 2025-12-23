# Fase 3: Rigor Teórico Avanzado - Documentación Matemática

## 📐 Conceptos Fundamentales

### Dualidad del Hipergrafo

La **dualidad** es una transformación fundamental en teoría de hipergrafos que intercambia el papel de nodos e hiperedges.

#### Definición Formal

Dado un hipergrafo $H = (V, E)$, su dual $H^* = (V^*, E^*)$ se define como:

$$V^* = E = \{e_1, e_2, ..., e_m\}$$
$$E^* = \{V_1^*, V_2^*, ..., V_n^*\} \text{ donde } V_i^* = \{e \in E : v_i \in e\}$$

Es decir:
- Cada nodo en $H^*$ corresponde a una hiperedge en $H$
- Cada hiperedge en $H^*$ corresponde a un nodo en $H$

#### Propiedades

1. **Doble Dual**: $(H^*)^* \approx H$ (estructura similar)
2. **Autoduales**: Algunos hipergrafos cumplen $H \cong H^*$
3. **Período**: Para hipergrafos finitos, existe $k$ tal que $(H^*)^k \cong H$

#### Invariantes bajo Dualidad

- Si $d(v)$ es el grado de nodo $v$ en $H$, entonces corresponde a $|e|$ en $H^*$
- La densidad se transforma según: $\rho(H^*) = \frac{|H|}{2^m}$

### Centralidad en Hipergrafos

**Centralidad** mide la importancia de nodos en la estructura de red.

#### 1. Centralidad de Grado

$$C_D(v) = \frac{deg(v)}{|E|}$$

Normalizado entre 0 y 1, donde $deg(v) = |\{e \in E : v \in e\}|$

#### 2. Centralidad Ponderada

$$C_W(v) = \sum_{e \in E, v \in e} w(e)$$

Suma de pesos de hiperedges que contienen al nodo.

#### 3. Betweenness Centrality

Para hipergrafos, adaptamos el concepto de rutas más cortas:

$$C_B(v) = \sum_{e_i \neq e_j} \frac{\sigma(e_i, v, e_j)}{\sigma(e_i, e_j)}$$

Donde $\sigma$ cuenta pares de hiperedges separados solo por el nodo $v$.

#### 4. Eigenvector Centrality

$$\lambda c_i = \sum_j A_{ij} c_j$$

Donde $\mathbf{c}$ es el eigenvector dominante de la matriz de adyacencia $A$.

Se calcula iterativamente (Power Iteration Method):

$$c^{(t+1)} = \frac{A \cdot c^{(t)}}{||A \cdot c^{(t)}||}$$

#### 5. Closeness Centrality

$$C_C(v) = \frac{n-1}{\sum_{u \neq v} d(v, u)}$$

Donde $d(v, u)$ es la distancia más corta entre nodos.

### Clustering en Hipergrafos

**Clustering** mide qué tan densamente conectados están los vecinos de un nodo.

#### Coeficiente de Clustering Local

Para un nodo $v$ con vecinos $N(v)$:

$$C_L(v) = \frac{\text{# de hiperedges entre vecinos}}{|N(v)| \cdot (|N(v)| - 1) / 2}$$

Retorna valor en $[0, 1]$.

#### Clustering Global (Transitivity)

$$C_G = \frac{\text{# triángulos cerrados}}{\text{# triángulos potenciales}}$$

Un triángulo está cerrado si sus tres nodos están todos conectados.

#### Índice de Homofilia

$$H = \frac{\text{# conexiones entre nodos del mismo tipo}}{\text{# conexiones totales}}$$

Mide la preferencia de nodos similares a conectarse.

#### Modularidad

$$Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$$

Donde:
- $A_{ij}$ es el peso entre nodos $i, j$
- $k_i$ es el grado del nodo $i$
- $c_i$ es la comunidad del nodo $i$
- $\delta$ es la función indicadora

### Propiedades Espectrales

**Propiedades espectrales** analizan eigenvalores de matrices asociadas.

#### 1. Matriz de Adyacencia

$$A_{ij} = \begin{cases} 1 & \text{si } \exists e \in E : i, j \in e \\ 0 & \text{otherwise} \end{cases}$$

#### 2. Matriz de Grados

$$D = \text{diag}(\deg(v_1), \deg(v_2), ..., \deg(v_n))$$

#### 3. Matriz Laplaciana

$$L = D - A$$

Propiedades:
- Siempre semidefinida positiva
- Eigenvalue mínimo es 0 (conectividad trivial)
- Segundo eigenvalue ($\lambda_2$) mide conectividad algebraica

#### 4. Matriz Laplaciana Normalizada

$$L_{norm} = I - D^{-1/2} A D^{-1/2}$$

Eigenvalues normalizados en $[0, 2]$.

#### 5. Energía Espectral

$$E(H) = \sum_{i=1}^{n} |\lambda_i|$$

Donde $\lambda_i$ son eigenvalues de la matriz de adyacencia.

#### 6. Spectral Gap (Algebraic Connectivity)

$$\lambda_2 = \text{segundo eigenvalue más pequeño de } L$$

Mide qué tan difícil es desconectar el hipergrafo. Valores altos indican alta conectividad.

#### 7. Índice de Wiener Espectral

$$W_E = \sum_{i < j} \frac{1}{1 + d(i,j)}$$

Donde $d(i,j)$ es distancia en el grafo de proximidad.

## 🔬 Implementación en HIPERGRAFO

### Dualidad

```typescript
const dual = DualidadHipergrafo.calcularDual(hipergrafo);
const esAutodual = DualidadHipergrafo.esAutodual(hipergrafo);
const periodo = DualidadHipergrafo.calcularPeriodoDualidad(hipergrafo);
```

Complejidad:
- Tiempo: $O(|V| \cdot |E|^2)$ para calcular dual
- Espacio: $O(|V| + |E|)$

### Centralidad

```typescript
// Grado: O(|E|)
const cent_d = CentralidadHipergrafo.centralidadGrado(hg, v);

// Betweenness: O(|E|^2)
const cent_b = CentralidadHipergrafo.centralidadBetweenness(hg, v);

// Eigenvector: O(k * n^2) donde k = iteraciones
const eigens = CentralidadHipergrafo.centralidadEigenvector(hg, 10);
```

### Clustering

```typescript
// Local: O(|N(v)|^2)
const coef_local = ClusteringHipergrafo.coeficienteClusteringLocal(hg, v);

// Global: O(n^3) para triplas de nodos
const coef_global = ClusteringHipergrafo.coeficienteClusteringGlobal(hg);

// Modularidad: O(|E|) con partición dada
const modularity = ClusteringHipergrafo.calcularModularidad(hg, partition);
```

### Propiedades Espectrales

```typescript
// Matriz Laplaciana: O(n^3) con métodos estándar
const L = PropiedadesEspectrales.calcularMatrizLaplacianaNormalizada(hg);

// Energía: O(n^3)
const energia = PropiedadesEspectrales.calcularEnergiaEspectral(hg);

// Spectral Gap: Aproximación O(n^2)
const gap = PropiedadesEspectrales.calcularGapEspectral(hg);
```

## 📊 Relaciones Matemáticas

### Proposición 1: Conectividad

Si $\lambda_2(L) > 0$, entonces $H$ es **conexo**.

**Prueba**: $\lambda_2 = 0$ iff existe descomposición no trivial de $V$.

### Proposición 2: Dualidad Preserva Conectividad

Si $H$ es conexo, $H^*$ no necesariamente lo es.

**Contraejemplo**: Estrella: un nodo central conectado a $n-1$ nodos aislados.

### Proposición 3: Autoduales de Orden Finito

Todo hipergrafo finito satisface $(H^*)^k \cong H$ para algún $k \leq 4$.

**Prueba**: Se basa en periodicidad de transformaciones de incidencia.

## 🎯 Interpretación de Resultados

### Alto Clustering Global
- Comunidades densas y bien definidas
- Relevante para redes neuronales: neuronas de capas similares

### Alto Spectral Gap
- Red muy conectada y robusta
- Difícil de particionar
- Buena propagación de información

### Baja Modularidad
- Comunidades débiles o inexistentes
- Red homogénea

### Autoduales
- Simetría perfecta nodo-hiperedge
- Raro en aplicaciones reales

## 📚 Referencias Teóricas

1. Bretto, A. (2013). "Hypergraph Theory: An Introduction"
2. Battiston, F., et al. (2014). "Structural measures for multiplex networks"
3. Estrada, E. (2012). "The Structure of Complex Networks"
4. Newman, M. E. J. (2010). "Networks: An Introduction"

---

**Última actualización**: Diciembre 2025
