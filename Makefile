# ============================================================================
# Makefile for K-Means Clustering Implementation
# Automates compilation, testing, and cleanup tasks
# ============================================================================

# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LIBS = -lm
PYTHON = python3

# Target executable
TARGET = kmeans
SOURCE = kmeans.c
PYTHON_SCRIPT = kmeans.py

# Default target
.PHONY: all
all: $(TARGET)

# Compile C program
$(TARGET): $(SOURCE)
	@echo "Compiling $(SOURCE) with strict ANSI C compliance..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)
	@echo "✓ Successfully compiled: $(TARGET)"

# Generate test data
.PHONY: test_data
test_data: sample_data.csv

sample_data.csv:
	@echo "Generating sample test data..."
	@$(PYTHON) << 'EOF'
import random
random.seed(42)

# Cluster 1: around (0, 0)
for _ in range(50):
    print(f"{random.gauss(0, 0.5):.2f},{random.gauss(0, 0.5):.2f}")

# Cluster 2: around (10, 10)
for _ in range(50):
    print(f"{random.gauss(10, 0.5):.2f},{random.gauss(10, 0.5):.2f}")

# Cluster 3: around (20, 0)
for _ in range(50):
    print(f"{random.gauss(20, 0.5):.2f},{random.gauss(0, 0.5):.2f}")
EOF
	@echo "✓ Sample data created: sample_data.csv"

# Run tests: compare C and Python implementations
.PHONY: test
test: $(TARGET) sample_data.csv
	@echo ""
	@echo "=========================================="
	@echo "Testing K-Means Implementation"
	@echo "=========================================="
	@echo ""
	@echo "📊 Running C version with K=3, max_iter=400..."
	@./$(TARGET) 3 400 < sample_data.csv > c_output.txt
	@echo "✓ C version output saved to c_output.txt"
	@echo ""
	@echo "🐍 Running Python version with K=3, max_iter=400..."
	@$(PYTHON) $(PYTHON_SCRIPT) 3 400 < sample_data.csv > python_output.txt
	@echo "✓ Python version output saved to python_output.txt"
	@echo ""
	@echo "Comparing outputs (should be very similar):"
	@echo "--- C Version Output ---"
	@cat c_output.txt
	@echo ""
	@echo "--- Python Version Output ---"
	@cat python_output.txt
	@echo ""
	@echo "✓ Test complete! (Note: Small numerical differences are expected)"

# Test error handling
.PHONY: test_errors
test_errors: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo "Testing Error Handling"
	@echo "=========================================="
	@echo ""
	@echo "Test 1: Invalid cluster count (K=0)"
	@echo "1,2,3" | ./$(TARGET) 0 2>/dev/null || echo "✓ Correctly rejected"
	@echo ""
	@echo "Test 2: Non-integer cluster count"
	@echo "1,2,3" | ./$(TARGET) abc 2>/dev/null || echo "✓ Correctly rejected"
	@echo ""
	@echo "Test 3: Invalid iteration count (iter=0)"
	@echo "1,2,3" | ./$(TARGET) 1 0 2>/dev/null || echo "✓ Correctly rejected"
	@echo ""
	@echo "Test 4: Iteration out of range (iter=1500)"
	@echo "1,2,3" | ./$(TARGET) 1 1500 2>/dev/null || echo "✓ Correctly rejected"
	@echo ""

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TARGET) $(TARGET).o
	@rm -f c_output.txt python_output.txt
	@echo "✓ Cleanup complete"

# Clean everything including test data
.PHONY: clean_all
clean_all: clean
	@echo "Removing test data..."
	@rm -f sample_data.csv
	@echo "✓ Full cleanup complete"

# Display help information
.PHONY: help
help:
	@echo ""
	@echo "=========================================="
	@echo "K-Means Clustering - Build System"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  make all        - Compile C implementation (default)"
	@echo "  make test       - Run full test suite comparing C and Python"
	@echo "  make test_data  - Generate sample test data"
	@echo "  make test_errors - Test error handling"
	@echo "  make clean      - Remove compiled binary and test output"
	@echo "  make clean_all  - Remove all generated files"
	@echo "  make help       - Display this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make              # Compile C version"
	@echo "  2. make test_data    # Generate sample data"
	@echo "  3. make test         # Run and compare implementations"
	@echo ""
	@echo "Manual usage:"
	@echo "  ./kmeans <K> [iterations] < data.csv"
	@echo "  python3 kmeans.py <K> [iterations] < data.csv"
	@echo ""

# Phony targets that don't produce files
.PHONY: all test test_data test_errors clean clean_all help
