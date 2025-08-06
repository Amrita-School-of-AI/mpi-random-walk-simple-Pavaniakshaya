#include <iostream>
#include <cstdlib> 
#include <ctime>   
#include <mpi.h>

// Function declarations for walker and controller processes
void walker_process();
void controller_process();

// Global variables shared across processes
int domain_size;    // Defines the boundary of the walk (e.g., -domain_size to +domain_size)
int max_steps;      // Maximum number of steps each walker can take
int world_rank;     // Rank (ID) of the process
int world_size;     // Total number of processes

int main(int argc, char **argv)
{
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank (ID) of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Expecting two arguments: <domain_size> and <max_steps>
    if (argc != 3)
    {
        // Only the controller (rank 0) prints the usage error
        if (world_rank == 0)
        {
            std::cerr << "Usage: mpirun -np <num_processes> " << argv[0]
                      << " <domain_size> <max_steps>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Convert command-line arguments to integers
    domain_size = std::atoi(argv[1]);
    max_steps = std::atoi(argv[2]);

    // Controller is assigned rank 0
    if (world_rank == 0)
    {
        controller_process();
    }
    else
    {
        // All other ranks are walkers
        walker_process();
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

// Function executed by each walker process
void walker_process()
{
    // Seed the random number generator using time and rank to ensure different seeds
    std::srand(static_cast<unsigned int>(std::time(NULL)) + world_rank);

    int position = 0;  // Walker starts at position 0
    int steps = 0;     // Steps taken so far

    // Walker continues until it exits the domain or reaches max_steps
    while (std::abs(position) <= domain_size && steps < max_steps)
    {
        // Randomly choose direction: -1 (left) or +1 (right)
        int direction = (std::rand() % 2 == 0) ? -1 : 1;
        position += direction;
        steps++;
    }

    // Report from walker
    std::cout << "Rank " << world_rank << ": Walker finished in " << steps << " steps." << std::endl;

    // Notify controller that this walker has completed
    int done_signal = 1;
    MPI_Send(&done_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

// Function executed by the controller process (rank 0)
void controller_process()
{
    int completed_walkers = 0;
    int total_walkers = world_size - 1;  // Exclude the controller itself

    // Loop until all walkers report completion
    while (completed_walkers < total_walkers)
    {
        MPI_Status status;

        // Wait for any incoming message from any walker
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int count;

        // Get number of items in the incoming message
        MPI_Get_count(&status, MPI_INT, &count);

        // Expecting 1 integer message (the done signal)
        if (count == 1)
        {
            int message;
            MPI_Recv(&message, count, MPI_INT,
                     status.MPI_SOURCE, status.MPI_TAG,
                     MPI_COMM_WORLD, &status);
            completed_walkers++;
        }
    }

    // Final report by controller
    std::cout << "Controller: All " << total_walkers << " walkers have finished." << std::endl;
}
