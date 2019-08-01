#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>

using namespace std;

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

double computeMeanS(const vector<int>& sizes) {

    if (sizes[1] == 0) return 1.;

    double numerator = 0.;
    double denominator = 0.;
    int s;
    int N = sizes.size();
    for (int i=0; i<N; i++) {
        s = sizes[i];
        numerator += s*s;
        denominator += s;
    }
    return numerator/denominator;
}

void computeMeasures(vector<int>& ptr, int& N1, int& N2, double& meanS) {
    int N = ptr.size();
    int EMPTY = -N-1;
    int r, n_clusters;
    vector<int> sizes;
    for (int i=0; i<N; i++) {
        r = ptr[i];
        if ((r < 0) && (r != EMPTY))
            sizes.push_back(-r);
    }
    n_clusters = sizes.size();
    if (sizes.size() == 1)
        sizes.push_back(0);
    if (sizes.size() == 2)
        sizes.push_back(0);

    vector<int>::iterator it;
    it = max_element(sizes.begin(), sizes.end());
    N1 = *it;
    iter_swap(it, sizes.rbegin());
    sizes.pop_back();
    it = max_element(sizes.begin(), sizes.end());
    N2 = *it;
    meanS = computeMeanS(sizes);
    
}

void writeData(ostream& out, int N1, int N2, double meanS) {
    out << N1 << " " << N2 << " " << meanS << endl;
}

int findroot(vector<int>& ptr, int i)
{
    if (ptr[i]<0) return i;
    return ptr[i] = findroot(ptr, ptr[i]);
}

void percolate(vector<set<int> >& nn_set, vector<int>& ptr, vector<int>& order, ostream& out)
{
  int i;
  int s1;
  int r1, r2;
  int N1, N2;
  double meanS;
  int N = order.size();
  int EMPTY = (-N-1);

  for (i=0; i<N; i++) ptr[i] = EMPTY;
  for (i=0; i<N; i++) {
    r1 = s1 = order[i];
    ptr[s1] = -1;
    for (int s2 : nn_set[s1]) {
      if (ptr[s2] != EMPTY) {
        r2 = findroot(ptr, s2);
        if (r2!=r1) {
            if (ptr[r1]>ptr[r2]) {
                ptr[r2] += ptr[r1];
                ptr[r1] = r2;
                r1 = r2;
            } else {
                ptr[r1] += ptr[r2];
                ptr[r2] = r1;
            }
        }
      }
    }
    computeMeasures(ptr, N1, N2, meanS);
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
