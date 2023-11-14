# linear-equation-solving-comparison

Implementation of linear equation solvers using different methods in python.
The project consists of a simple matrix class, two iterative methods of solving linear equations (Jacobi, Gauss-Seidl) and one direct method (LU decomposition).
The main script calculates the execution time of the provided methods and generates different plots made for the report required for the assignment.

The matrix class could be improved in terms of performance, as the main data structurev it's based on is a python list.
This could be changed to a numpy array for faster operations. It wasn't changed in the project, as the current implementation was sufficient for the assignment's requirements.
The main task was to compare execution times, and the difference wouldn't change the ratio between different methods' execution times.

In the repository there's also included a report alongside its LaTeX source.

Made as an university assignment for "Numerical Methods" class.

## Dependencies

- `matplotlib >= 3.6.2`
