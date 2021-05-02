## LU DECOMPOSITION

- How to run?
    - On terminal, run  `g++ -o <nome_ficheiro> LU_Sequential.cpp`
    - `./nome_ficheiro`
    - Introduce size of matrix (square matrix, so L x L)
- What it does?
    - Given a size _n_, generates a random matrix, and stores it into `original_matrix.csv`
    - Given a matrix, size _n_, calculates LU decomposition, and then splits the result matrix into the L and U matrices, storing each of them and also teh result_matrix in `csv` files