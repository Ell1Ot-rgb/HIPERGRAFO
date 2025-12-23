# HIPERGRAFO - AI Agent Instructions

## Project Overview
**HIPERGRAFO** is a node network system ("RED DE NODOS"). This is an early-stage project focused on building a graph/network data structure with node relationships and interactions.

## Architecture Principles
- **Node-Based Design**: Core functionality revolves around nodes as first-class entities that can connect and communicate
- **Graph Structure**: Expect hierarchical or networked relationships between components
- **Scalability**: Design with network-scale problems in mind (many interconnected nodes)

## Development Guidelines

### Before Adding Features
1. Consider how new features affect node connectivity and data flow
2. Ensure additions maintain separation between node logic and network topology
3. Document any new node types or relationship patterns introduced

### Code Organization
When the project expands, follow this structure:
- `src/core/` - Core node and graph abstractions
- `src/nodes/` - Specific node type implementations
- `tests/` - Test suite matching source structure
- `docs/` - Architecture and design documentation

### Naming Conventions
- Node classes: PascalCase (e.g., `DataNode`, `ProcessingNode`)
- Network/graph operations: camelCase (e.g., `connectNodes`, `traverseNetwork`)
- Constants: UPPER_SNAKE_CASE for configuration values

### Testing
- Each node implementation should have unit tests
- Integration tests for network behaviors (multi-node interactions)
- Include graph traversal and connection tests

## Key Decisions & Rationale
- **Node-Centric**: Building around node abstraction allows independent unit testing and reusability
- **Red de Nodos**: Emphasis on networked relationships, not isolated components

## Requesting Clarification
When expanding this project, provide:
- Node types you're implementing and their responsibilities
- Expected connection patterns (one-to-many, cyclic, etc.)
- Data flow requirements through the network
