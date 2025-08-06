[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/m56AXb1r)
# C++ & MPI: Parallel Random Walk 🚶‍♂️

This assignment challenges you to build a parallel random walk simulation using C++ and the Message Passing Interface (MPI). The goal is to simulate multiple "walkers" exploring a 1D path simultaneously, with each walker running on a different process.

### The Random Walk Concept

A random walk is a path made of a series of random steps. In this assignment, our walkers live on a 1D line. They start at position 0 and at each step, they randomly move one unit to the left (-1) or to the right (+1). A walk ends when a walker steps outside a predefined domain, for example, a domain from -20 to +20.

### Parallelizing with MPI

We will use MPI to parallelize this simulation. When you run an MPI program with `P` processes, one process (rank 0) will act as a **controller**, and the other `P-1` processes will be the **walkers**.

Communication is key. Walkers need to inform the controller when they have finished their walk. MPI's point-to-point communication functions, `MPI_Send` and `MPI_Recv`, are perfect for this.

```cpp
// Example: Sending an integer from process 1 to process 0
int data_to_send = 100;
int destination_rank = 0;
int tag = 0;

// On the sending process (e.g., a walker with rank > 0)
MPI_Send(
    &data_to_send,      // Pointer to the data to send
    1,                  // Number of elements to send
    MPI_INT,            // Data type of the elements
    destination_rank,   // Rank of the destination process
    tag,                // A message tag (like a message type)
    MPI_COMM_WORLD      // The communicator
);

// On the receiving process (e.g., the controller with rank 0)
int received_data;
int source_rank = 1;
MPI_Status status; // To store information about the received message

MPI_Recv(
    &received_data,     // Pointer to the buffer for incoming data
    1,                  // Max number of elements to receive
    MPI_INT,            // Data type of the elements
    source_rank,        // Rank of the source process
    tag,                // The message tag to look for
    MPI_COMM_WORLD,     // The communicator
    &status             // Pointer to the status object
);
```

---

## Task 🎯

You must implement a parallel random walk simulation in the provided `random_walk.cpp` file. Your program will be launched from the command line and receive arguments specifying the simulation parameters.

### Process Roles:

- **Process 0 (Controller):** This process does not simulate a walk. Its job is to wait for each walker process to send a message indicating it has finished. After hearing back from all walkers, it should print a final summary.
- **Processes 1 to P-1 (Walkers):** Each of these processes simulates a single random walk.

### Walker Logic:

1.  A walker starts at position 0.
2.  In a loop, it randomly decides to move left (-1) or right (+1) and updates its position.
3.  The walk is **finished** if the walker's position goes outside the domain `[-D, D]`, where `D` is the domain size given as a command-line argument.
4.  When its walk is finished, the walker must:
    - Print a message to `stdout` that includes the word **"finished"**. The autograder depends on this keyword.
    - Send a message to the controller (rank 0) to signal that it's done.

**Note on Random Numbers:** To generate random numbers, use `rand()`. It's good practice to seed the random number generator once per process using `srand(time(NULL) * rank)`. This ensures each walker follows a different random path.

---

## Input Format

Your program will receive its parameters from the command line. It will be launched like this:
`mpirun -np <num_processes> ./random_walk <domain_size> <max_steps>`

- `<num_processes>`: The total number of processes MPI will launch.
- `<domain_size>`: An integer `D` defining the walk domain `[-D, D]`.
- `<max_steps>`: An integer safety limit to prevent infinite loops. A walk should also end if it reaches this many steps.

These arguments can be accessed in your C++ code via `argv`. For example, `argv[1]` will be the domain size. Remember to convert them from strings to integers using `atoi()`.

---

## Output Format

For autograding to work, your program's output must follow this format:

1.  Each **walker** process must print exactly one line containing the word **"finished"** when its walk is complete.
2.  The **controller** process should print "Controller: All X walkers have finished.". It can print other summary information after all walkers are done.

### Sample Run

**Command:**

```shell
mpirun -np 5 ./random_walk 20 10000
```

This command runs 1 controller and 4 walkers. The domain is `[-20, 20]`.

**Sample Output:**
_(Note: The order and number of steps will vary with every run due to randomness)_

```
Rank 3: Walker finished in 152 steps.
Rank 1: Walker finished in 44 steps.
Rank 4: Walker finished in 380 steps.
Rank 2: Walker finished in 88 steps.
Controller: All 4 walkers have finished..
```

Good luck\! 👍
