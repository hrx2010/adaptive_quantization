#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
using namespace std;

// command line parameters
int num_layers = 55;
int total_num_data = 9252800; //32342720;//30498752;
string filename_data_points = "all_resnet_50_pareto_condition_data_wei_act.txt";

// definitions
struct Node{
    int layer, dead_zone, quant;
    float length;
    float error;
};

vector<Node> *data_points;

float eps = 1e-12;

void solve(vector<int> &solutions_dead_zone , vector<int> &solutions_quant , float target){
    for(int i = 0 ; i < num_layers ; i ++){
        solutions_dead_zone.push_back(0);
        solutions_quant.push_back(0);
    }

    float left = -100000000.0;
    float right = 0.0;

    while(right - left >= eps){
        float total_size = 0.0;
        float mid = (left + right) * 0.5;

        for(int i = 0 ; i < num_layers ; i ++){
            int selected = -1;
            float intercept = 100000000;

            for(int j = 0 ; j < data_points[i].size() ; j ++){
                float p = data_points[i][j].length;
                float q = data_points[i][j].error;
                float b = q - mid * p;

                if(b < intercept || selected == -1){
                    intercept = b;
                    selected = j;
                }
            }

            solutions_dead_zone[i] = data_points[i][selected].dead_zone;
            solutions_quant[i] = data_points[i][selected].quant;
            total_size += data_points[i][selected].length;
            //printf("------------- %d\n" , selected);
        }

        printf("@ %f %f %f\n"  , left , right , total_size);

        if(total_size > target) right = mid;
        else left = mid;
    }
}

int main(){
    data_points = new vector<Node>[num_layers];

    FILE *file_data_points = fopen(filename_data_points.c_str() , "r");
    for(int i = 0 ; i < num_layers ; i ++){
        int m;
        fscanf(file_data_points , "%d" , &m);

        for(int j = 0 ; j < m ; j ++){
            int layer, dead_zone, quant;
            float length;
            float error;

            fscanf(file_data_points , "%d %d %d %f %f\n" , &layer , &dead_zone , &quant , &length , &error);

            Node node;
            node.layer = layer, node.dead_zone = dead_zone, node.quant = quant, node.length = length , node.error = error;

            data_points[i].push_back(node);
        }
    }

    for(int i = 15 ; i <= 160 ; i ++){
        vector<int> solutions_dead_zone;
        vector<int> solutions_quant;
        solve(solutions_dead_zone , solutions_quant , 1.0 * i / 10.0 * total_num_data);
        char filename[1010];
        sprintf(filename , "bit_allocations_%d.txt" , i);

        FILE *out = fopen(filename , "w");
        for(int i = 0 ; i < solutions_dead_zone.size() ; i ++){
            fprintf(out , "%d %d\n" , solutions_dead_zone[i] , solutions_quant[i]);
            printf("%d %d\n" , solutions_dead_zone[i] , solutions_quant[i]);
        }
        fclose(out);

        printf("finish %d\n" , i);
    }
    fclose(file_data_points);

    return 0;
}
