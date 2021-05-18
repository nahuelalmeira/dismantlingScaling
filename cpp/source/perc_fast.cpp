/* 

Compile as:

g++ -o2 -std=c++11 ../source/perc_v2.cpp -o ../perc_v2

*/

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
  int N1, N2;
  int num, denom, n_comps, large, small, prev_N1;
  double meanS;
  int N = order.size();
  int EMPTY = (-N-1);
  bool overpass, new_gcc;
  num = denom = n_comps = 0;
  for (i=0; i<N; i++) ptr[i] = EMPTY;
  for (i=0; i<N; i++) {
    r1 = s1 = order[i];
    ptr[s1] = -1;
    num += 1;
    denom += 1;
    n_comps += 1;
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
    if (meanS == 0) meanS = 1.0;
    }
    writeData(out, N1, 0, meanS);
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
