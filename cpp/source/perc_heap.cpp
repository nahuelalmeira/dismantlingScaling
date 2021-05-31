/* 

Compile as:

g++ -o2 -std=c++11 ../source/perc_heap.cpp -o ../perc_heap

*/

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

using namespace std;


// Data structure to store a max-heap node
struct PriorityQueue
{
private:
	// vector to store heap elements
	vector<int> A;

	// return parent of `A[i]`
	// don't call this function if `i` is already a root node
	int PARENT(int i) {
		return (i - 1) / 2;
	}

	// return left child of `A[i]`
	int LEFT(int i) {
		return (2*i + 1);
	}

	// return right child of `A[i]`
	int RIGHT(int i) {
		return (2*i + 2);
	}

	// Recursive heapify-down algorithm.
	// The node at index `i` and its two direct children
	// violates the heap property
	void heapify_down(int i)
	{
		// get left and right child of node at index `i`
		int left = LEFT(i);
		int right = RIGHT(i);

		int largest = i;

		// compare `A[i]` with its left and right child
		// and find the largest value
		if (left < size() && A[left] > A[i]) {
			largest = left;
		}

		if (right < size() && A[right] > A[largest]) {
			largest = right;
		}

		// swap with a child having greater value and
		// call heapify-down on the child
		if (largest != i)
		{
			swap(A[i], A[largest]);
			heapify_down(largest);
		}
	}

	// Recursive heapify-up algorithm
	void heapify_up(int i)
	{
		// check if the node at index `i` and its parent violate the heap property
		if (i && A[PARENT(i)] < A[i])
		{
			// swap the two if heap property is violated
			swap(A[i], A[PARENT(i)]);

			// call heapify-up on the parent
			heapify_up(PARENT(i));
		}
	}

public:
	// return size of the heap
	unsigned int size() {
		return A.size();
	}

	// Function to check if the heap is empty or not
	bool empty() {
		return size() == 0;
	}

	// insert key into the heap
	void push(int key)
	{
		// insert a new element at the end of the vector
		A.push_back(key);

		// get element index and call heapify-up procedure
		int index = size() - 1;
		heapify_up(index);
	}

	// Function to remove an element with the highest priority (present at the root)
	void pop()
	{
		try {
			// if the heap has no elements, throw an exception
			if (size() == 0)
			{
				throw out_of_range("Vector<X>::at() : "
						"index is out of range(Heap underflow)");
			}

			// replace the root of the heap with the last element
			// of the vector
			A[0] = A.back();
			A.pop_back();

			// call heapify-down on the root node
			heapify_down(0);
		}
		// catch and print the exception
		catch (const out_of_range &oor) {
			cout << endl << oor.what();
		}
	}

	// Function to return an element with the highest priority (present at the root)
	int top()
	{
		try {
			// if the heap has no elements, throw an exception
			if (size() == 0)
			{
				throw out_of_range("Vector<X>::at() : "
						"index is out of range(Heap underflow)");
			}

			// otherwise, return the top (first) element
			return A.at(0);		// or return A[0];
		}
		// catch and print the exception
		catch (const out_of_range &oor) {
			cout << endl << oor.what();
		}
	}
};

int loadOrder(const string file_name, vector<int>& order, bool verbose) {

  ifstream file;
  int node, N=0;
    /* First loop to get N and M */
    file.open(file_name);
    if(!file.is_open()) {
        cerr << "Could not open file '" << file_name << "'" << endl;
        assert(0);
    }

    while(true) {
        file >> node;
        if(file.eof()) break;
        //cout << node << endl;
        //getchar();
        order.push_back(node);
        N++;
    }
    reverse(order.begin(), order.end());
    if (verbose) cout << "N = " << N << endl;
    file.close();
    return N;
}

void loadNeighbors(const string file_name, vector<set<int> >& nn_set, bool verbose) {
    ifstream file;
    int a, b;
    if(verbose) cout << "Adding data to graph" << endl;
    /* Second loop to add data to graph */
    file.open(file_name, ios::in);
    while(true) {
        file >> a >> b;
        if(file.eof()) break;
        //cout << a << " " << b << endl;
        if(a==b) continue;
        nn_set[a].insert(b); nn_set[b].insert(a);
    }
    file.close();
}

void writeData(ostream& out, int N1, int N2, double meanS) {
    out << N1 << " " << N2 << " " << meanS << endl;
}

int findroot(vector<int>& ptr, int i)
{
    if (ptr[i]<0) return i;
    return ptr[i] = findroot(ptr, ptr[i]);
}

void percolate(
    vector<set<int> >& nn_set, 
    vector<int>& ptr, 
    vector<int>& order, 
    ostream& out
)
{
  int i;
  int s1;
  int r1, r2;
  int N1 = 0, N2 = 0;
  int mN1;
  bool found;
  int size, heap_size;
  int num, denom, n_comps, large, small, prev_N1;
  double meanS;
  int N = order.size();
  int EMPTY = (-N-1);
  bool overpass, new_gcc;
  num = denom = n_comps = 0;

  PriorityQueue heap;
  int sizes[N+1] = {};

  for (i=0; i<N; i++) ptr[i] = EMPTY;
  for (i=0; i<N; i++) {
    r1 = s1 = order[i];
    ptr[s1] = -1;
    num += 1;
    denom += 1;
    n_comps += 1;
    heap.push(1);
    sizes[1]++;
    for (int s2 : nn_set[s1]) {
      overpass = new_gcc = false;
      if (ptr[s2] != EMPTY) {
        r2 = findroot(ptr, s2);
        if (r2!=r1) {
            if (ptr[r1]>ptr[r2]) {
                large = -ptr[r2];
                small = -ptr[r1];
                ptr[r2] += ptr[r1];
                ptr[r1] = r2;
                r1 = r2;
            } else {
                large = -ptr[r1];
                small = -ptr[r2];
                ptr[r1] += ptr[r2];
                ptr[r2] = r1;
            }

            if (sizes[small]) sizes[small]--;
            if (sizes[large]) sizes[large]--;
            sizes[small+large]++;
            heap.push(small+large);

            if (-ptr[r1] > N1) {
                new_gcc = true;
                if (large < N1) overpass = true;
                prev_N1 = N1;
                N1 = -ptr[r1];
            }
            if (new_gcc) {
                if (overpass) {
                    num = (
                        num - small*small - large*large 
                        + prev_N1*prev_N1
                    );
                    denom = denom - small - large + prev_N1;
                } else {
                    num = num - small*small;
                    denom = denom - small;
                }
            } else {
                num = (
                    num - small*small - large*large 
                    + (small+large)*(small+large)
                );
            }
        }
      }

    if (denom == 0) meanS = 0.0;
    else            meanS = float(num)/denom;
    if (n_comps == 1) num = denom = meanS = 0;
    if (meanS < 1e-6) meanS = 1.0;
    }

    if (heap.size() < 2) N2 = 0;
    else {
        mN1 = heap.top();
        heap.pop();
        heap_size = heap.size();
        for (int k=0; k < heap_size; k++) {
            size = heap.top();
            heap.pop();
            found = false;
            if (sizes[size] > 0) {
                N2 = size;
                heap.push(N2);
                heap.push(mN1);
                found = true;
                break;
            }
        }
        if (!found) {
            heap.push(mN1);
            N2 = 0;
        }
    }
    writeData(out, N1, N2, meanS);
    //printf("%i %i %i %f\n",i+1, N1, N2, meanS);
    //getchar();
  }
}

int main(int argc, char* argv[])
{
    int N;
    vector<int> ptr;
    vector<int> order;
    vector<set<int> > nn_set;

    string network = argv[1];
    string order_file = argv[2];
    string output_file = argv[3];
    ofstream output;
    output.open(output_file);

    //cout << "Before loadOrder" << endl;
    N = loadOrder(order_file, order, false);
    //cout << "N = " << N << endl;
    ptr.resize(N);
    nn_set.resize(N);
    for(int i=0; i<N; i++) nn_set[i].clear();

    //cout << "Before loadNeighbors" << endl;
    loadNeighbors(network, nn_set, false);
    //out << "Before percolate" << endl;
    percolate(nn_set, ptr, order, output);

    output.close();

    return 0;
}
