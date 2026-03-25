<div align="center">

# Clustering Algorithms Repository

Comprehensive exploration of three clustering algorithms (Lloyd's K-Means, K-Means++, SymNMF) from foundations to advanced techniques. 5.5x performance optimization, hybrid Python-C architecture, and comparative analysis.

![CI](https://img.shields.io/github/actions/workflow/status/odeliyach/Clustering-Algorithms-Lab/ci.yml?label=CI&color=BAFFC9&style=flat-square&logo=github-actions&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-BAFFC9?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![C](https://img.shields.io/badge/C-ANSI%20C99-BAFFC9)
![Speedup](https://img.shields.io/badge/Speedup-5.5x-pink)
</div>

---

## Overview

This repository contains three implementations exploring clustering algorithms and their optimization:

- **01-KMeans-Basic**: Pure C and Python implementation of Lloyd's algorithm
- **02-KMeans-Optimized**: Reduced convergence time by 3.6× (47 → 13 iterations) as measured by iteration count on 1 000-point synthetic datasets, by implementing K-Means++ weighted-probability initialization with a Python-C hybrid architecture
- **03-SymNMF-Advanced**: Improved clustering quality by 14% (Silhouette 0.42 → 0.48) as measured on graph-structured benchmarks, by implementing Symmetric Non-negative Matrix Factorization in C with a CPython C-API bridge

Each stage builds on the previous, demonstrating algorithm fundamentals, optimization techniques, and when to use advanced approaches.

---

## 📚 Project Structure

```
Clustering-Algorithms-Lab/
│
├── 01-KMeans-Basic/
│   ├── lloyd_clustering.c        # Pure ANSI C implementation
│   ├── lloyd_clustering.py       # Pure Python equivalent
│   ├── Makefile
│   └── README.md
│
├── 02-KMeans-Optimized/
│   ├── src/
│   │   ├── algorithm.py
│   │   ├── visualizers.py
│   │   └── utils.py
│   ├── ext/
│   │   ├── clustering.h
│   │   ├── clustering.c
│   │   └── clustering_module.c
│   ├── setup.py
│   ├── Makefile
│   └── README.md
│
├── 03-SymNMF-Advanced/
│   ├── src/
│   │   ├── symnmf.py
│   │   ├── analysis.py
│   │   └── __init__.py
│   ├── ext/
│   │   ├── symnmf.h
│   │   ├── symnmf.c
│   │   └── symnmfmodule.c
│   ├── setup.py
│   ├── Makefile
│   └── README.md
│
├── benchmark-suite/
│   └── compare_all.py
│
│
├── ANALYSIS.md
├── LICENSE
└── README.md (this file)
```

---

## Algorithm Comparison

| Aspect | K-Means Basic | K-Means++ | SymNMF |
|--------|---|---|---|
| **Speed** | 1.0x | 5.5x | 0.5x |
| **Quality** | 0.42 | 0.44 | 0.48 |
| **Memory** | 1.0x | 1.0x | 8.4x |
| **Best For** | Learning | Production | Graphs |

---

## 🎓 Learning Progression

### Phase 1: Foundations (01-KMeans-Basic)
- Lloyd's algorithm from scratch
- Pure C and Python
- Time: 3-5 hours

### Phase 2: Optimization (02-KMeans-Optimized)
- K-Means++ initialization
- Python-C hybrid
- 5.5x speedup
- Time: 8-10 hours

### Phase 3: Advanced (03-SymNMF-Advanced)
- Spectral clustering
- Matrix factorization
- Time: 10-15 hours

---

## 📊 Key Metrics

- **K-Means++ Speedup**: 5.5x faster than naive
- **Convergence**: 3.6x fewer iterations (47 → 13)
- **SymNMF Quality**: 14% better than K-Means on graph data
- **Documentation**: 5000+ lines

---

## 🚀 Quick Start

### K-Means Basic
```bash
cd 01-KMeans-Basic
gcc -ansi -Wall -Wextra -Werror -pedantic-errors -o lloyd lloyd_clustering.c -lm
```

### K-Means Optimized
```bash
cd 02-KMeans-Optimized
make build
```

### SymNMF Advanced
```bash
cd 03-SymNMF-Advanced
make build
python3 src/symnmf.py 5 symnmf data.csv
```

---

## 🧪 Testing

Run the full test suite with [pytest](https://docs.pytest.org/):

```bash
pip install pytest numpy
pytest tests/ -v
```

---

## 🐳 Docker

Build and run all tests inside a container:

```bash
docker build -t clustering-lab .
docker run --rm clustering-lab
```

---

## ⚙️ CI/CD

Every push and pull request triggers a GitHub Actions workflow that:

1. Compiles the Phase 1 C binary with strict warnings
2. Runs the full pytest suite

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details.

---

## 📖 Documentation

- **01-KMeans-Basic/README.md**: Algorithm fundamentals
- **02-KMeans-Optimized/README.md**: Optimization & hybrid architecture
- **03-SymNMF-Advanced/README.md**: Spectral clustering details
- **ANALYSIS.md**: Deep comparative analysis

---

## 📊 Repository Stats

- **Total Code**: 2000+ lines (Python + C)
- **Implementations**: 3 distinct algorithms
- **Languages**: Python, C, Makefile
- **Documentation**: Comprehensive READMEs + analysis
- **Benchmarks**: Comparative performance data

---

**Happy clustering! 🎉**
---

<div align="center">

Built by **Odeliya Charitonova** · [GitHub](https://github.com/odeliyach) · [LinkedIn](https://linkedin.com/in/odeliya-charitonova)

*Computer Science student @ Tel Aviv University, School of CS & AI*

</div>
