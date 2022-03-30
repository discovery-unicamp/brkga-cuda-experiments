#include"stdio.h"
#include"stdlib.h"
#include <math.h>
#include <assert.h> 
#include"string.h"
#include"time.h"
#include"float.h"
/* Calculates Euclidean distance between two nodes */
__host__ __device__ float distD(int i, int j, float *x, float*y)
{
	float dx = x[i] - x[j];
	float dy = y[i] - y[j]; 
	return(sqrtf( (dx * dx) + (dy * dy) ));
}
/* Maintains customer list holding its id, demand, and distance from depot */
struct nearest_neighbors
{
	int id, dmd;
	float dst;
	struct nearest_neighbors *next;
};
/* Prints m routes with its capacity and traveling distance */
void display_routes(int *k_routes,int *rt_st, int k, int *cap, int *rt_nodes, float* rt_dst, float dst)
{
	int x, curr, cnt ;
	for(x = 0; x < k; x++)
	{
		curr = rt_st[x];
		printf("\nroute %d: 0 ",x);
		cnt = 0;
		while(cnt < rt_nodes[x] )
		{
			printf(", %d",curr);
			curr = k_routes[curr];
			cnt++;
		}
		printf(", 0 ");
		printf("\tcap: %d rt_dst: %.02f nodes: %d",cap[x], rt_dst[x], rt_nodes[x]);
	}
	printf("\nFeasible solution cost: %.2f\n", dst);
}
/* swap heuristic is applied after feasible solution is constructed */
struct swap_data
{
	int rt_1, rt_2, ct_1, ct_2;
	float change_1, change_2;
	long tot_change;
};
/* A kernel function for swap heuristic for neighborhood generation and its cost computation */
__global__ void inter_swap_kernel(int*k_routes, int*rt_st, float*posx, float*posy, int * demands, int * cap, int Q, int n, int *rt_nodes, swap_data * sd, int *rt_id, int k)
{
	int gap1, gap2, x, j, rt_1;
	int prev_id_1, next_id_1;
	int prev_id_2, next_id_2;
	
	long change = 0;
	float change1, change2;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > 0 && i < n)
	{	
		rt_1 = rt_id[i];
		gap1 = demands[i] + Q - cap[rt_1];
		if(rt_nodes[rt_1] > 1)
		{
			prev_id_1 = 0;
			j = rt_st[rt_1];
			if(i == j)
				next_id_1 = k_routes[j];
			else
			{
				prev_id_1 = j;
				j = k_routes[j];
				while(j != i)
				{
					prev_id_1 = j;
					j = k_routes[j];
				}
				next_id_1 = k_routes[j];
			}
			for(j = rt_1 + 1; j < k; j++)
			{
				if(rt_nodes[j] > 1)
				{
					prev_id_2 = 0;
					x = rt_st[j];
					while(x != 0)
					{
						next_id_2 = k_routes[x];
						gap2 = demands[x] + Q - cap[j];
						if(demands[i] <= gap2 && demands[x] <= gap1)
						{
							change1 =  distD(prev_id_1, x, posx, posy)
								+ distD(x, next_id_1, posx, posy)
								- ( distD(prev_id_1, i, posx, posy) 
								+ distD(i, next_id_1, posx, posy));

							change2 =  distD(prev_id_2, i, posx, posy)
								+ distD(i, next_id_2, posx, posy)
								- (distD(prev_id_2, x, posx, posy) 
								+ distD(x, next_id_2, posx, posy));
							change = (change1 + change2) * 100; 
							if(change < sd[i].tot_change)
							{
								sd[i].rt_1 = rt_1;
								sd[i].rt_2 = j;
								sd[i].ct_1 = i;
								sd[i].ct_2 = x;
								sd[i].change_1 = change1;
								sd[i].change_2 = change2;
								sd[i].tot_change = change;
							}
						}
						prev_id_2 = x;
						x = k_routes[x];
					}
				}
			}
		}
	}
}
/* Finds the best solution after applying swap heuristcs */
void find_min_cpu(struct swap_data * sd, long sol)
{
	int min_i = 0, flag = 0;
	long minD = sd[0].tot_change;
	for(int i = 1; i < sol; i++)
	{
		if(sd[i].tot_change < minD)
		{
			minD = sd[i].tot_change;
			min_i = i;
			flag = 1;
		}
	}
	if(flag)
	{
		sd[0].tot_change = sd[min_i].tot_change;
		sd[0].change_1 = sd[min_i].change_1;
		sd[0].change_2 = sd[min_i].change_2;
		sd[0].ct_1 = sd[min_i].ct_1;
		sd[0].ct_2 = sd[min_i].ct_2;
		sd[0].rt_1 = sd[min_i].rt_1;
		sd[0].rt_2 = sd[min_i].rt_2;
	}
}
/* A kernel fucntion to initialize swap data structure */
__global__ void init_swap(struct swap_data *td, int n)
{
int i = threadIdx.x + blockIdx.x * blockDim.x ;
if(i < n)
	td[i].tot_change = 0;

}
/* This function is meant to distribute work over GPU */
float inter_swap(int*k_routes, int*rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int *demands, int Q, int *cap, int*rt_id, int n)
{
	int flag = 1;
	int thrd, blk;
	if(n < 128)
	{
		blk = 1;
		thrd = n;
	}
	else
	{
		thrd = 128;
		blk = (n - 1)/ thrd + 1;
	}

	struct swap_data * sd;
	cudaMallocManaged(&sd, sizeof(struct swap_data )* n);
	init_swap<<<blk, thrd>>>(sd, n);
	cudaDeviceSynchronize();
	int *d_k_routes, *d_rt_st, *d_rt_nodes, *d_cap, *d_rt_id;
	float *d_rt_dst;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_cap, sizeof(int) * k);
	cudaMalloc(&d_rt_id, sizeof(int) * n);
	cudaMalloc(&d_rt_dst, sizeof(float) * k);
	while(flag)
	{
		flag = 0;
		cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_cap, cap, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_dst, rt_dst, sizeof(float)* k, cudaMemcpyHostToDevice);

		inter_swap_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, demands, d_cap, Q, n, d_rt_nodes, sd, d_rt_id, k);
		cudaDeviceSynchronize();
		
		find_min_cpu(sd, n);
		if(sd[0].tot_change < 0)
		{
			cap[sd[0].rt_1] = cap[sd[0].rt_1] - demands[sd[0].ct_1] + demands[sd[0].ct_2];
			cap[sd[0].rt_2] = cap[sd[0].rt_2] - demands[sd[0].ct_2] + demands[sd[0].ct_1];
			rt_dst[sd[0].rt_1] += sd[0].change_1;
			rt_dst[sd[0].rt_2] += sd[0].change_2;
			rt_id[sd[0].ct_2] = sd[0].rt_1;
			rt_id[sd[0].ct_1] = sd[0].rt_2;
			dst += (float)sd[0].tot_change / 100;

			int next1, next2, prev1, prev2;
			int value = rt_st[sd[0].rt_2];
			while(value != sd[0].ct_2 && value != 0)
			{
				prev2 = value;
				value = k_routes[value];
			}
			value = rt_st[sd[0].rt_1];
			while(value != sd[0].ct_1 && value != 0)
			{
				prev1 = value;
				value = k_routes[value];
			}
			next1 = k_routes[sd[0].ct_1];
			next2 = k_routes[sd[0].ct_2];
			if(sd[0].ct_1 == rt_st[sd[0].rt_1])
			{
				rt_st[sd[0].rt_1] = sd[0].ct_2;
				k_routes[sd[0].ct_2] = next1;
			}
			else
			{
				k_routes[prev1] = sd[0].ct_2;
				k_routes[sd[0].ct_2] = next1;
			}
			if(sd[0].ct_2 == rt_st[sd[0].rt_2])
			{
				rt_st[sd[0].rt_2] = sd[0].ct_1;
				k_routes[sd[0].ct_1] = next2;
			}
			else
			{
				k_routes[prev2] = sd[0].ct_1;
				k_routes[sd[0].ct_1] = next2;
			}
			flag = 1;
			init_swap<<<blk, thrd>>>(sd, n);
			cudaDeviceSynchronize();
		}
	}
cudaFree(sd);
cudaFree(d_k_routes);
cudaFree(d_rt_st);
cudaFree(d_rt_nodes);
cudaFree(d_rt_dst);
cudaFree(d_cap);
cudaFree(d_rt_id);
return dst;
}

/* A function calculates an effect of removing a customer from one route and adding it to other route */
float add_diff(int *k_routes, int st, int j, float *posx, float *posy, int nodes,float rt_dst, int *best_nbr )
{
	float min_dst = INFINITY, dst, change1;
	int curr_nbr, next_nbr;
	if(st == 0)
		change1 = distD(0, j, posx, posy) * 2;
	else if(nodes >= 2)
	{
		while(st != 0)
		{
			dst = distD(st, j, posx, posy);
			if(dst < min_dst)
			{
				min_dst = dst; 
				curr_nbr = st;		
				*best_nbr = st;
				next_nbr = k_routes[st] == 0 ? 0: k_routes[st];
			}
			st = k_routes[st];
		}
		change1 = distD(curr_nbr, j, posx, posy)
			+ distD(next_nbr, j, posx, posy)
			- distD(curr_nbr, next_nbr, posx, posy);
	}
	else
	{
		change1 = distD(st, j, posx, posy)
			+ distD(0, j, posx, posy)
			- distD(0, st, posx, posy);
		*best_nbr = st;
	}
	return change1;
}
/* relocate heuristic is applied after feasible solution is constructed */
struct relocate_data
{
	int rt_1, rt_2, ct_1, ct_2;
	float change_1, change_2;
	long tot_change;
};
/* A device function calculates an effect of removing a customer from one route and adding it to other route */
__device__ float add_diff(int *k_routes, int *rt_st, int j, float *posx, float *posy, int nodes, int *best_nbr, int rt_2)
{
	float change1;
	float cst, min_dst = INFINITY;
	int y, next_nbr, curr_nbr;
	if(nodes == 0)
	{
		change1 = distD(0, j, posx, posy) * 2;
		*best_nbr = 0;
	}
	else if(nodes == 1)
	{
		change1 = distD(rt_st[rt_2], j, posx, posy)
			+ distD(0, j, posx, posy)
			- distD(0, rt_st[rt_2], posx, posy);
		*best_nbr = rt_st[rt_2];
	}
	else
	{
		y = rt_st[rt_2];
		while(y != 0)
		{
			cst = distD(y, j, posx, posy);
			if(cst < min_dst)
			{
				min_dst = cst; 
				curr_nbr = y;		
				next_nbr = k_routes[y];
			}
			y = k_routes[y];
		}
		change1 = distD(curr_nbr, j, posx, posy)
			+ distD(next_nbr, j, posx, posy)
			- distD(curr_nbr, next_nbr, posx, posy);
		*best_nbr = curr_nbr;
	}
	return change1;
}
/* This function adds a customer to specified route */
void itr_add_cust(int *k_routes, int *rt_st, int st, int add, int i, int nbr)
{
	int next;
	if(st == 0)
		rt_st[i] = add;
	else
	{
		next = k_routes[nbr];
		k_routes[nbr] = add;
		k_routes[add] = next;
	}
}
/* This function removes a customer from specified route */
void itr_rm_cust(int *k_routes, int *rt_st, int st, int j, int min_j)
{
	int prev;
	if(st == min_j)
	{
		if(k_routes[st] != 0)
			rt_st[j] = k_routes[st];
		else
			rt_st[j] = 0;
	}
	else
	{
		while(st != min_j)
		{
			prev = st;
			st = k_routes[st];
		}
		k_routes[prev] = k_routes[min_j];
	}
}
/* A GPU kernel for relocate heuristic */
__global__ void relocate_kernel(int *k_routes, int *rt_st, float*posx, float*posy, int * demands, int * cap, int Q, int n, int *rt_nodes, struct relocate_data * sd, int *rt_id, int k)
{
	int gap,j, rt_1;
	int prev_id_1, next_id_1;
	int *nbr;
	nbr = (int *)malloc(sizeof(int));
	long change;
	float change1, change2;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > 0 && i < n)
	{	
		rt_1 = rt_id[i];
		if(rt_nodes[rt_1] > 1)
		{
			prev_id_1 = 0;
			j = rt_st[rt_1];
			if(j == i)
			{
				prev_id_1 = 0;
				next_id_1 = k_routes[j];
			}
			else
			{
				while(j != i)
				{
					prev_id_1 = j;
					j = k_routes[j];
				}
				next_id_1 = k_routes[i];
			}
			change2 = distD(next_id_1, prev_id_1, posx, posy)
	 			- distD(prev_id_1, i, posx, posy) 
				- distD(next_id_1, i, posx, posy);
			for(j = rt_1+1; j < k; j++)
			{
				gap = Q - cap[j];
				if(demands[i] <= gap)
				{
					change1 = add_diff(k_routes, rt_st, i, posx, posy, rt_nodes[j], nbr, j);
					change = (change1 + change2) * 100; 
					if(change < sd[i].tot_change)
					{
						sd[i].rt_1 = rt_1;
						sd[i].rt_2 = j;
						sd[i].ct_1 = i;
						sd[i].ct_2 = *nbr;
						sd[i].change_1 = change1;
						sd[i].change_2 = change2;
						sd[i].tot_change = change;
					}
				}
			}
		}
	}
	free(nbr);
}
/* This function finds best performing solution after applying relocate heuristic */
void find_min_relocate(struct relocate_data * sd, long sol)
{
	int min_i = 0, flag = 0;
	long minD = sd[0].tot_change;
	for(int i = 1; i < sol; i++)
	{
		if(sd[i].tot_change < minD)
		{
			minD = sd[i].tot_change;
			min_i = i;
			flag = 1;
		}
	}
	if(flag)
	{
		sd[0].tot_change = sd[min_i].tot_change;
		sd[0].change_1 = sd[min_i].change_1;
		sd[0].change_2 = sd[min_i].change_2;
		sd[0].ct_1 = sd[min_i].ct_1;
		sd[0].ct_2 = sd[min_i].ct_2;
		sd[0].rt_1 = sd[min_i].rt_1;
		sd[0].rt_2 = sd[min_i].rt_2;
	}
}
/*A kernel function is meant to initialize relocate data structure */
__global__ void init_relocate(struct relocate_data *td, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	if(i < n)
		td[i].tot_change = 0;
}
/* this function is made to distribute work of relocate heuristic over GPU  */
float relocate(int *k_routes, int *rt_st, float *rt_dst, float *posx, float *posy, int k, float dst, int *demands, int Q, int *cap, int*rt_id, int n, int *rt_nodes)
{
	int flag = 1;
	int thrd, blk;
	if(n < 128)
	{
		blk = 1;
		thrd = n;
	}
	else
	{
		thrd = 128;
		blk = (n - 1)/ thrd + 1;
	}
	struct relocate_data * sd;
	cudaMallocManaged(&sd, sizeof(struct relocate_data )* n);
	init_relocate<<<blk, thrd>>>(sd, n);
	cudaDeviceSynchronize();
	int *d_k_routes, *d_rt_st, *d_rt_nodes, *d_cap, *d_rt_id;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_cap, sizeof(int) * k);
	cudaMalloc(&d_rt_id, sizeof(int) * n);
	while(flag)
	{
		flag = 0;
		cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_cap, cap, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);
		relocate_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, demands, d_cap, Q, n, d_rt_nodes, sd, d_rt_id, k);
		cudaDeviceSynchronize();

		find_min_relocate(sd, n);
		if(sd[0].tot_change < 0)
		{
			cap[sd[0].rt_1] = cap[sd[0].rt_1] - demands[sd[0].ct_1] ;
			cap[sd[0].rt_2] = cap[sd[0].rt_2] + demands[sd[0].ct_1];
			rt_dst[sd[0].rt_1] += sd[0].change_2;
			rt_dst[sd[0].rt_2] += sd[0].change_1;
			rt_id[sd[0].ct_1] = sd[0].rt_2;
			dst += ((float)sd[0].tot_change/100);
			if(rt_st[sd[0].rt_2] == 0)
				rt_nodes[sd[0].rt_2]++;
			itr_rm_cust(k_routes, rt_st, rt_st[sd[0].rt_1], sd[0].rt_1, sd[0].ct_1);
			itr_add_cust(k_routes, rt_st, rt_st[sd[0].rt_2], sd[0].ct_1, sd[0].rt_2, sd[0].ct_2);
				
			rt_nodes[sd[0].rt_1]--;
			rt_id[sd[0].ct_1] = sd[0].rt_2;
			rt_nodes[sd[0].rt_2]++;
			if(rt_st[sd[0].rt_1] == 0)
			{
				rt_nodes[sd[0].rt_1] = 0;
				cap[sd[0].rt_1] = 0;
				rt_dst[sd[0].rt_1] = 0;
			}
			flag = 1;
			init_relocate<<<blk, thrd>>>(sd, n);
			cudaDeviceSynchronize();
		}
	}
	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_cap);
	cudaFree(d_rt_id);
	cudaFree(sd);
	return dst;
}

struct threeOptData
{
	int x, y, z, rt;
	long change;
};
/* This function finds best solution from 3-opt data structure */
int find_min_three(struct threeOptData*td, struct threeOptData*kBest, int *rt_id, long n)
{
	int flag = 0;
	for(int i = 1; i < n; i++)
	{
		if(td[i].change < kBest[rt_id[i]].change)
		{
			kBest[rt_id[i]].change = td[i].change;
			kBest[rt_id[i]].x = td[i].x;
			kBest[rt_id[i]].y = td[i].y;
			kBest[rt_id[i]].z = td[i].z;
			kBest[rt_id[i]].rt = td[i].rt;
			flag = 1;
		}
	}
	return flag;
}
/* A kernel function to perform 3-opt on GPU */
__global__ void three_opt_kernel(int*k_routes, int*rt_st, float*posx, float*posy, long n, struct threeOptData *td, int *rt_nodes, int *rt_id)
{
	long change = 0;
	int next_x, next_y, next_z, rt_1, i, y, z;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x > 0 && x < n)
	{	
		rt_1 = rt_id[x];
		if(rt_nodes[rt_1] > 3)
		{
			i = rt_st[rt_1];
			if(x == i)
				next_x = k_routes[i];
			else
				next_x = k_routes[x];
			y = next_x;
			while(y != 0)
			{
				next_y = k_routes[y];
				z = next_y;
				while(z != 0)
				{
					next_z = k_routes[z];
					change = 100 * (distD(x, z, posx, posy) 
						+ distD(next_x, next_y, posx, posy) 
						+ distD(y, next_z, posx, posy) 
						- ( distD(x, next_x, posx, posy)
						+ distD(y, next_y, posx, posy)
						+ distD(z, next_z, posx, posy)));

					if(change < td[x].change)
					{
						td[x].rt = rt_1;
						td[x].x = x;
						td[x].y = y;
						td[x].z = z;
						td[x].change = change;
					}
					z = k_routes[z];
				}
				y = k_routes[y];
			}
		}
	}
}
/* Kernel function to initialize 3-opt data structure */
__global__ void init_three(struct threeOptData *td, int n)
{
int i = threadIdx.x + blockIdx.x * blockDim.x ;
if(i < n)
	td[i].change = 0;

}
/* This function is made to distribute 3-opt heuristic work over GPU */
float three_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n, int * rt_id)
{
	int i, j_next, i_next, next, prev, m_next;
	int thrd, blk;
	struct threeOptData *td;
	cudaMallocManaged(&td, sizeof(struct threeOptData)*n);

	if(n < 128)
	{
		blk = 1;
		thrd = n;
	}
	else
	{
		thrd = 128;
		blk = (n - 1) / thrd + 1;
	}
	init_three<<<blk, thrd>>>(td, n);
	cudaDeviceSynchronize();
	int *d_k_routes, *d_rt_st, *d_rt_nodes, *d_rt_id;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_id, sizeof(int) * n);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

	three_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
	cudaDeviceSynchronize();
	struct threeOptData *kBest = (struct threeOptData *)malloc(sizeof(struct threeOptData )* k);
	for (i = 0; i< k; i++)
		kBest[i].change = 0;
	int flag = find_min_three(td, kBest, rt_id, n);
	while( flag )
	{
		for(int r = 0; r < k; r++)
		{
			if(kBest[r].change < 0)
			{
				rt_dst[r] +=  ((float)kBest[r].change/100);
				dst = dst + ((float)kBest[r].change/100);
				i_next = kBest[r].x == 0? rt_st[r] : k_routes[kBest[r].x];
				j_next = kBest[r].y == 0? rt_st[r] : k_routes[kBest[r].y];
				m_next = kBest[r].z == 0? rt_st[r] : k_routes[kBest[r].z];
				if(j_next != kBest[r].z)
				{
					next = k_routes[j_next];
					k_routes[j_next] = i_next;
					prev = j_next;
					while(next != kBest[r].z)
					{
						i = k_routes[next];
						k_routes[next] = prev;
						prev = next;
						next = i;
					}
					k_routes[kBest[r].z] = prev;
					if(kBest[r].x)
						k_routes[kBest[r].x] = kBest[r].z;
					else
						rt_st[r] = kBest[r].z;
					k_routes[kBest[r].y] = m_next;
				}
				else
				{
					k_routes[j_next] = i_next;
					if(kBest[r].x)
						k_routes[kBest[r].x] = kBest[r].z;
					else
						rt_st[r] = kBest[r].z;
					k_routes[kBest[r].y] = m_next;
				}
			}
		}
		init_three<<<blk, thrd>>>(td, n);
		cudaDeviceSynchronize();

		cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

		three_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
		cudaDeviceSynchronize();
		for (i = 0; i< k; i++)
			kBest[i].change = 0;
		flag = find_min_three(td, kBest, rt_id, n);
	}

	cudaFree(td);
	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_rt_id);

	free(kBest);
	return dst; 
}

struct twoOptData
{
	int x, y, rt;
	long change;
};
/* This function finds the best solution after applying 2-opt heuristic */
int find_min_two(struct twoOptData*td, struct twoOptData* kBest, int *rt_id, long n)
{
	int flag = 0;
	for(int i = 1; i < n; i++)
	{

		if(td[i].change < kBest[rt_id[i]].change)
		{
			kBest[rt_id[i]].change = td[i].change;
			kBest[rt_id[i]].x = td[i].x;
			kBest[rt_id[i]].y = td[i].y;
			kBest[rt_id[i]].rt = td[i].rt;
			flag = 1;
		}
	}
	return flag;
}
/* A kernel function is made to perform 2-opt heuristic work over GPU */
__global__ void two_opt_kernel(int*k_routes, int*rt_st, float*posx, float*posy, long n, struct twoOptData *td, int *rt_nodes, int *rt_id)
{
	register long change = 0;
	float change1;
	int next_id_1, next_id_2, rt_1, j, x, x_n, y, y_n;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > 0 && i < n )
	{	
		rt_1 = rt_id[i];
		if(rt_nodes[rt_1] > 2)
		{
			j = rt_st[rt_1];
			if(i == j)
			{
				next_id_1 = k_routes[j];
				x = 0;
				x_n = i;
				y = i;
				while(y)
				{
					y_n = k_routes[y];
					change1 = distD(x, y, posx, posy)
						 + distD(x_n, y_n, posx, posy)
						 - ( distD(x, x_n, posx, posy) 
						 + distD(y, y_n, posx, posy));
					change = change1 * 100; 
					if(change < td[i].change)
					{
						td[i].rt = rt_1;
						td[i].x = x;
						td[i].y = y;
						td[i].change = change;
					}
					y = k_routes[y];
				}
			}
			else
				next_id_1 = k_routes[i];
			while(j)
			{
				if(i != j)
				{
					next_id_2 = k_routes[j];
					change1 = distD(i, j, posx, posy)
						 + distD(next_id_1, next_id_2, posx, posy)
						 - ( distD(i, next_id_1, posx, posy) 
						 + distD(j, next_id_2, posx, posy));
					change = change1 * 100; 
					if(change < td[i].change)
					{
						td[i].rt = rt_1;
						td[i].x = i;
						td[i].y = j;
						td[i].change = change;
					}
				}
				j = k_routes[j];
			}
		}
	}
}
/* A kernel function is to initialize 2-opt data structure */
__global__ void init_two(struct twoOptData *td, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	if(i < n)
		td[i].change = 0;
}
/* This function distribute 2-opt heuristic work over GPU */
float two_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n, int * rt_id)
{
	int j_next, i_next, next, prev, vst;
	int thrd, blk;
	struct twoOptData *td;
	cudaMallocManaged(&td, sizeof(struct twoOptData)*n);
	if(n < 128)
	{
		blk = 1;
		thrd = n;
	}
	else
	{
		thrd = 128;
		blk = (n - 1) / thrd + 1;
	}
	init_two<<<blk, thrd>>>(td, n);
	cudaDeviceSynchronize();
	int *d_k_routes, *d_rt_st, *d_rt_nodes, *d_rt_id;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_id, sizeof(int) * n);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

	two_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
	cudaDeviceSynchronize();
	struct twoOptData* kBest = (struct twoOptData*) malloc(sizeof(struct twoOptData )* k);
	for(int i = 0; i < k; i++)
		kBest[i].change = 0;

	int flag = find_min_two(td, kBest, rt_id, n);
	while(flag)
	{
		for(int i = 0; i < k; i++)
		{
			if(kBest[i].change < 0)
			{
				rt_dst[i] +=  ((float)kBest[i].change/100);
				dst += ((float)kBest[i].change/100);
				if(kBest[i].x == 0 || kBest[i].y == 0)
				{
					i_next = rt_st[i];
					if(kBest[i].x == 0)
					{
						rt_st[i] = kBest[i].y;
						j_next = k_routes[kBest[i].y];
						vst = kBest[i].y;
					}
					else
					{
						rt_st[i] = kBest[i].x;
						j_next = k_routes[kBest[i].x];
						vst = kBest[i].x;
					}
				}
				else
				{
					int x = rt_st[i];
					while(x != kBest[i].x && x != kBest[i].y)
						x= k_routes[x];
					if(x == kBest[i].x)
					{
						i_next = k_routes[x];
						k_routes[x] = kBest[i].y;
						j_next = k_routes[kBest[i].y];
						vst = kBest[i].y;
					}
					else
					{
						i_next = k_routes[x];
						k_routes[x] = kBest[i].x;
						j_next = k_routes[kBest[i].x];
						vst = kBest[i].x;
					}

				}
				while(vst != i_next )
				{
					prev = i_next;
					next = k_routes[i_next];

					while(next != vst)
					{
						prev = next;
						next = k_routes[next];
					}
					k_routes[next] = prev;
					vst = prev;
				}	
				k_routes[i_next] = j_next;

			}
		}
		init_two<<<blk, thrd>>>(td, n);
		cudaDeviceSynchronize();

		cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

		two_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
		cudaDeviceSynchronize();

		for(int i = 0; i < k; i++)
			kBest[i].change = 0;

		flag = find_min_two(td, kBest, rt_id, n);
	}
	cudaFree(td);
	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_rt_id);

	free(kBest);
	return dst; 
}
/* A data structure is created for or-opt heuristic to hold each thread's neighborhood generation information */
struct orOptData
{
	int x, y, rt, stride;
	long change;
};
/* This function finds the best performing solution after applying or-opt */ 
int find_min_or(struct orOptData*td, struct orOptData*kBest, int *rt_id, long n)
{
	int flag = 0;
	for(int i = 1; i < n; i++)
	{
		if(td[i].change < kBest[rt_id[i]].change)
		{
			kBest[rt_id[i]].change = td[i].change;
			kBest[rt_id[i]].x = td[i].x;
			kBest[rt_id[i]].y = td[i].y;
			kBest[rt_id[i]].rt = td[i].rt;
			kBest[rt_id[i]].stride = td[i].stride;
			flag = 1;
		}
	}
	return flag;
}
/* A kernel function is made for or-opt to work over GPU  */
__global__ void or_opt_kernel(int*k_routes, int*rt_st, float*posx, float*posy, long n, struct orOptData *td, int *rt_nodes, int *rt_id)
{
	register long change = 0;
	float change1;
	int i_prev, rt_1, j, x, stride, j_next;
	int count1, count2, last, last_next, curr2;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > 0 && i < n)
	{	
		rt_1 = rt_id[i];
		if(rt_nodes[rt_1] > 2)
		{
			count1 = 0;
			j = rt_st[rt_1];
			if(i == j)
				i_prev = 0;
			else
			{
				x = k_routes[j];
				count1++;
				while(x != i)
				{
					i_prev = x;
					x = k_routes[x];
					count1++;
				}
			}

			if(rt_nodes[rt_1] == 3)
				stride = 2;
			else
				stride = 3;
			for(; stride > 0; stride--)
			{
				if(count1 < (rt_nodes[rt_1] - 1 - stride))
				{
					j = i; 
					for(count2 = 0; count2 < (stride - 1); count2++)
						j = k_routes[j];
					last = j;
					last_next = k_routes[last];
					curr2 = last_next;
					for(count2 = count1 + stride; count2 < (rt_nodes[rt_1] - 1); count2++)
					{
						j_next = k_routes[curr2];
						change1 = distD(curr2, i, posx, posy)
							+ distD(j_next, last, posx, posy)
							+ distD(i_prev, last_next, posx, posy)
							- ( distD(i_prev, i, posx, posy)
							+ distD(last, last_next, posx, posy)
							+ distD(j_next, curr2, posx, posy));
						change = change1 * 100;

						if(change < td[i].change)
						{
							td[i].change = change;
							td[i].x = i;
							td[i].y = curr2;
							td[i].stride = stride;
							td[i].rt = rt_1;
						}
						curr2 = j_next;
					}
				}
			}
		}
	}
}
/*A kernel function is meant to initialize or data structure */
__global__ void init_or(struct orOptData *td, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	if(i < n)
		td[i].change = 0;
}
/* This function distribute work of or-opt over GPU */
float or_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n, int * rt_id)
{
	int i, next, i_prev;
	int thrd, blk;
	int j, j_prev, y;
	struct orOptData *td;
	cudaMallocManaged(&td, sizeof(struct orOptData)*n);
	if(n < 128)
	{
		blk = 1;
		thrd = n;
	}
	else
	{
		thrd = 128;
		blk = (n - 1) / thrd + 1;
	}
	init_or<<<blk, thrd>>>(td, n);
	cudaDeviceSynchronize();
	int *d_k_routes, *d_rt_st, *d_rt_nodes, *d_rt_id;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_id, sizeof(int) * n);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

	or_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
	cudaDeviceSynchronize();

	struct orOptData * kBest = (struct orOptData *) malloc(sizeof(struct orOptData )* k);
	for(int i = 0; i < k; i++)
		kBest[i].change = 0;

	int flag = find_min_or(td, kBest, rt_id, n);
	while( flag )
	{
		for(int r = 0; r < k; r++)
		{
			if(kBest[r].change < 0)
			{
				rt_dst[r] +=  ((float)kBest[r].change/100);
				dst = dst + ((float)kBest[r].change/100);
				i = rt_st[r];		
				while(i != kBest[r].x && i != kBest[r].y)
				{
					i_prev = i;
					i = k_routes[i];
				}
				int xory = i == kBest[r].x ? 0 : 1;
				j = 0; y = i;
				while( j < kBest[r].stride)
				{
					j_prev = y;
					y = k_routes[y];
					j++;
				}
				if(kBest[r].x == rt_st[r] || kBest[r].y == rt_st[r])
				{
					if(kBest[r].x == rt_st[r])
					{
						rt_st[r] = y;
						while(y != kBest[r].y)
							y = k_routes[y];
					}
					else
					{
						rt_st[r] = y;
						while(y != kBest[r].x)
							y = k_routes[y];
					}
				}
				else
				{
					if(xory == 0)
					{
						k_routes[i_prev] = y;
						while(y != kBest[r].y)
							y = k_routes[y];
					}
					else
					{
						k_routes[i_prev] = y;
						while(y != kBest[r].x)
							y = k_routes[y];
					}
				}
				next = k_routes[y];
				k_routes[y] = i;
				k_routes[j_prev] = next;
			}
		}
		init_or<<<blk, thrd>>>(td, n);
		cudaDeviceSynchronize();

		cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rt_id, rt_id, sizeof(int)* n, cudaMemcpyHostToDevice);

		or_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_st, posx, posy, n, td, d_rt_nodes, d_rt_id);
		cudaDeviceSynchronize();
		for(int i = 0; i < k; i++)
			kBest[i].change = 0;

		flag = find_min_or(td, kBest, rt_id, n);

	}
	cudaFree(td);
	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_rt_id);

	free(kBest);
	return dst; 
}

int route_join(int *k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, int *demands, int Q, int *cap, int * rt_id)
{
	int i, j, x, y;
	float d1, d2;
	for(i = 0; i < k; i++ )
	{
		for(j = i+1; j < k; j++ )
		{
			if(cap[i] + cap[j] <= Q )
			{
				x = rt_st[i];
				while(k_routes[x] != 0)
					x = k_routes[x];
				d1 = rt_dst[i] - distD(x, 0, posx, posy);
			
				y = rt_st[j];
				d2 = rt_dst[j] - distD(y, 0, posx, posy);
				//
				k_routes[x] = y;
				// update distance
				rt_dst[i] = d1 + d2 + distD(x, y, posx, posy);
				// update rt_id
				while(y !=  0 )
				{
					rt_id[y] = i;
					y = k_routes[y];
				}
				// upadte rt_nodes & cap
				rt_nodes[i] += rt_nodes[j];
				cap[i] += cap[j];

				// Move rt_st, rt_dst, rt_nodes,  cap by 1
				for(x = j; x < k-1; x++)
				{
					rt_st[x] = rt_st[x + 1];
					rt_dst[x] = rt_dst[x + 1];
					rt_nodes[x] = rt_nodes[x + 1];
					cap[x] = cap[x + 1];
				}
				for(x = j; x < k-1; x++)
				{
					y = rt_st[x];
					while(y != 0)
					{
						rt_id[y] = x;
						y = k_routes[y];
					}
				}

				k--;
				j = i;
			}
		}
	}
	return k;
}
/* This function takes care of solution construction and improvement phases */
void create_soln(float *posx, float *posy, int *demands, int n, int k, int Q, FILE *f, float opt)
{
	int x, flg;
	int *v, *cap, *rt_nodes;
	float dst = 0, *rt_dst, rate;
	int *k_routes, *rt_st;
	float d1 ;int *rt_id;
	clock_t start, end, start1, end1;

	k_routes = (int*)malloc(sizeof(int) * n);
	rt_st = (int*)calloc(k, sizeof(int));
	v = (int*)calloc(n, sizeof(int));
	cap = (int*)calloc(k, sizeof(int));
	rt_id = (int*)calloc(n, sizeof(int));
	rt_nodes = (int*)calloc(k, sizeof(int));
	rt_dst = (float*)calloc(k, sizeof(float));

	start = clock();
	int cnt = 1, i = 1, dmd, j, y;
	int * unvst;
	struct nearest_neighbors *first= NULL, *last, *crt, *crt1;
	unvst = (int*)calloc(n, sizeof(int));
	for(i = 1; i < n; i++)
	{
		dst = distD(0, i, posx, posy);
		crt = (struct nearest_neighbors *) malloc(sizeof(struct nearest_neighbors));
		crt->id = i;
		crt->dst = dst;
		crt->dmd = demands[i];
		crt->next = NULL;
		if(first == NULL)
		{
			first = crt;
			last = crt;
		}
		else
		{
			last->next = crt;
			last = crt;
		}
	}
	crt = first; 
	while(crt->next != NULL)
	{
		crt1 = crt->next;
		while(crt1 != NULL)
		{
			if(crt1->dst > crt->dst )
			{
				dst = crt->dst;
				i = crt->id;
				dmd = crt->dmd;
				crt->dst = crt1->dst;
				crt->id = crt1->id;
				crt->dmd = crt1->dmd;
				crt1->dst = dst;
				crt1->id = i;
				crt1->dmd = dmd;
			}
			else if(crt1->dst == crt->dst && crt1->dmd < crt->dmd)
			{
				dst = crt->dst;
				i = crt->id;
				dmd = crt->dmd;
				crt->dst = crt1->dst;
				crt->id = crt1->id;
				crt->dmd = crt1->dmd;
				crt1->dst = dst;
				crt1->id = i;
				crt1->dmd = dmd;
			}
			crt1 = crt1->next;
		}
		crt = crt->next;
	}

//-------------------- Finishes creating farthest_neighbor structure ------------------ 
//---------------Forming routes for each vehicle considering all constraints-------
	i = 0;
	crt = first;
	cnt = 0;
	int r, cur_i, min_i, min_iy, old_k = 0;
	float min_d;
	dst = 0;
	x = 0; y = 1;
	for(r = 0; r < k; r++)
	{
		back:
		while( crt!=NULL)
		{
			if(v[crt->id])
				crt = crt->next;
			else
				break;
		}
		if(crt == NULL) break;
		cnt++;
		rt_nodes[r]++;
		cap[r] += demands[crt->id];
		rt_dst[r] += crt->dst * 2;
		dst += rt_dst[r];
		rt_id[crt->id] = r;
		rt_st[r] = crt->id;
		k_routes[crt->id] = 0;
		cur_i = rt_st[r];
		j = 0;
		if(x == 0)
		{
			unvst[x] = crt->id;
			unvst[crt->id] = -1;
			x++;
			if(unvst[x] == -1)
				x++;
		}
		else
		{
			if(crt->id >= x)
			{
				if(v[x] == 0)
				{
					unvst[x] = crt->id;
					unvst[crt->id] = x;
					x++;
					if(unvst[x] == -1)
						x++;
				}
				else
				{
					int tmp = unvst[x];
					unvst[x] = crt->id;
					unvst[crt->id] = tmp;
					x++;
					if(unvst[x] == -1)
						x++;
				}
			}
			else
			{
				min_iy = unvst[crt->id];
				if(min_iy < x)
				{
					i = x;
					min_iy = unvst[i];
					while(min_iy != crt->id && i < n)
					{	min_iy = unvst[i];
						i++;
					}
					min_iy = i-1;
				}
				if(v[x] == 0)
				{
					unvst[x] = crt->id;
					unvst[min_iy] = x;
					x++;
					if(unvst[x] == -1)
						x++;
				}
				else
				{
					int tmp = unvst[x];
					unvst[x] = crt->id;
					unvst[min_iy] = tmp;
					x++;
					if(unvst[x] == -1)
						x++;
				}
			}
		}
		v[crt->id] = 1;
		do{
			min_d = DBL_MAX;
			for(y = x; y < n; y++)
			{
				if(rt_st[r] != y && !v[y])
				{
					d1 = distD(cur_i, y, posx, posy) 
					+ distD(0, y, posx, posy) 
					- distD(cur_i, 0, posx, posy);
					if(d1 < min_d)
					{
						min_d = d1;
						min_i = y;
					}
				}
				else if(v[y] && unvst[y] > 0)
				{
					d1 = distD(cur_i, unvst[y], posx, posy) 
					+ distD(0, unvst[y], posx, posy) 
					- distD(cur_i, 0, posx, posy);
					if(d1 < min_d)
					{
						min_d = d1;
						min_i = unvst[y];
						min_iy = y;
					}
				}
			}
			if( (min_d < DBL_MAX) && (cap[r] + demands[min_i] <= Q))
			{
				k_routes[cur_i] = min_i;
				k_routes[min_i] = 0;
				rt_dst[r] += min_d;
				dst += min_d;

				rt_id[min_i] = r;
				cap[r] += demands[min_i];
				if(v[x] == 0)
				{
					unvst[x] = min_i;
					if(v[min_i] == 0 && min_i >= x)
						unvst[min_i] = x;
					else
						unvst[min_iy] = x;
					x++;
					if(unvst[x] == -1)
						x++;
				}
				else
				{
					int tmp = unvst[x];
					unvst[x] = min_i;
					if(v[min_i] == 0 && min_i >= x)
						unvst[min_i] = tmp;
					else
						unvst[min_iy] = tmp;
					x++;
					if(unvst[x] == -1)
						x++;
				}
				v[min_i] = 1;
				cur_i = min_i;
				rt_nodes[r]++;
				cnt++;
				j++;
			}
			else
				break;

		}while(cap[r] <= Q);
		
	}
	if(cnt < n-1)
	{
		if(old_k == 0)
			old_k = k;
		r = k;
		k++;
		rt_st = (int *) realloc(rt_st, sizeof(int) * k);
		cap= (int *) realloc(cap, sizeof(int) * k);
		rt_nodes= (int *) realloc(rt_nodes, sizeof(int) * k);
		rt_dst= (float *) realloc(rt_dst, sizeof(float) * k);
		rt_st[r] = 0;
		rt_nodes[r] = 0;
		rt_dst[r] = 0;
		cap[r] = 0;
		goto back;
	}
	d1 = 0;
	for(x = 0; x < k; x++)
	{
		if(rt_st[x] != 0)
			d1 += rt_dst[x];
	}
	if(old_k < k)
		k = route_join(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, demands, Q, cap, rt_id);
	d1 = 0;
	for(x = 0; x < k; x++)
	{
		if(rt_st[x] != 0)
			d1 += rt_dst[x];
	}
	end = clock();	
	printf("%.2f\t %f\t", d1, ((double) (end - start)) / CLOCKS_PER_SEC);	
	dst = d1;
	flg = 1;
	start1 = clock();
	while(flg)
	{
		flg = 0;
		d1 = inter_swap(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, dst, demands, Q, cap, rt_id, n);
		d1 = relocate(k_routes, rt_st, rt_dst, posx, posy, k, d1, demands, Q, cap, rt_id, n, rt_nodes);
		if (d1 < dst)
		{
			d1 = two_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n, rt_id);
			d1 = or_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n, rt_id);
			d1 = three_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n, rt_id);
			dst = d1;
			flg = 1;
		}

	}
	end1 = clock();
	printf("%.2f\t",dst);
	rate = (d1 - opt) / opt * 100;
	printf("%.2f\t%.2f\t%f\n", opt, rate, ((double) (end1 - start1)) / CLOCKS_PER_SEC);

	free(v);
	free(cap);
	free(rt_id);
	free(rt_nodes);
	free(rt_dst);
	free(rt_st);
}

int main(int argc, char *argv[])
{
	float *posx, *posy, opt, len;
	int *demands;
	char str[256];  
	int n, k, Q;
	int ch, cnt, in1, i=0;
	float in2, in3;
	FILE *f;
	f = fopen(argv[1], "r");
	if (f == NULL) {fprintf(stderr, "could not open file \n");  exit(-1);}
	char* p = strstr(argv[1], "GoldOne");
	if(p)
	{
		fscanf(f, "%s %f\n", str, &opt);
		while(strcmp(str, "COMMENT:") != 0)
			fscanf(f, "%s %f\n", str, &opt);

		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "DIMENSION:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "CAPACITY:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		Q = i;
		fscanf(f, "%s %f\n", str, &len);
		while(strcmp(str, "DISTANCE:") != 0)
			fscanf(f, "%s %f\n", str, &len);

		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "VEHICLES:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		k = i;
		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		posx = (float *)malloc(sizeof(float) * n);  
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[in1] = in2;
			posy[in1] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[in1] = in2;
			cnt++;
		}
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%f %f\n", &in2, &in3);
		posx[0] = in2;
		posy[0] = in3;
		demands[0] = 0;
		printf("%s\t",argv[1]);

		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "GoldTwo"))
	{
		fscanf(f, "%s %f\n", str, &opt);
		while(strcmp(str, "COMMENT:") != 0)
			fscanf(f, "%s %f\n", str, &opt);

		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "DIMENSION:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "CAPACITY:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		Q = i;

		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "VEHICLES:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		k = i;
		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		posx = (float *)malloc(sizeof(float) * n);  
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[in1] = in2;
			posy[in1] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[in1] = in2;
			cnt++;
		}
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%f %f\n", &in2, &in3);
		posx[0] = in2;
		posy[0] = in3;
		demands[0] = 0;
		printf("%s\t",argv[1]);

		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "kyto"))
	{
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		n = atoi(str);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		Q = atoi(str);

		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		k = atoi(str);

		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[cnt] = in2;
			posy[cnt] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		int dmd = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[cnt] = in2;
			dmd += in2;
			cnt++;
		}
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%f %f\n", &in2, &in3);
		printf("%s\t",argv[1]);

		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "bel") )
	{
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		n = atoi(str);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		Q = atoi(str);
		fscanf(f, "%s\n", str);

		if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[cnt] = in2;
			posy[cnt] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		int dmd = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[cnt] = in2;
			dmd += in2;
			cnt++;
		}
		k = ceil((float)dmd / Q);
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		printf("%s\t",argv[1]);
		create_soln(posx, posy, demands, n, k, Q, f, 0);
	}

	else if(strstr(argv[1], "A") || strstr(argv[1], "E") || strstr(argv[1], "B") || strstr(argv[1], "F") || strstr(argv[1], "M")|| strstr(argv[1], "P"))
	{
		ch = getc(f);  while ((ch != EOF) && (ch != 'n')) ch = getc(f);
		ch = getc(f);  
		while ((ch != EOF) && (ch != '-'))
		{
			str[i++] = ch; 
			ch = getc(f);
		}
		str[i]='\0';
		n = atoi(str);

		ch = getc(f);  while ((ch != EOF) && (ch != 'k')) ch = getc(f);
		fscanf(f, "%s\n", str);
		k = atoi(str);
		char buf[10];
		fscanf(f, "%s", buf);
		if (strstr(argv[1], "M-n200-k17") == NULL && strstr(argv[1], "M-n200-k16") == NULL )
		{
			while(  strcmp(buf, "value:") != 0 )
				fscanf(f, "%s", buf);
			fscanf(f, "%f", &opt);
		}		
		else
		{
			if(strstr(argv[1], "M-n200-k17"))
				opt = 1275;
			else
				opt = 1274;
		}
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		Q = atoi(str);
		fscanf(f, "%s\n", str);

		if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[cnt] = in2;
			posy[cnt] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[cnt] = in2;
			cnt++;
		}
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%s", str);
		printf("%s\t",argv[1]);

		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "Li") )
	{
		fscanf(f, "%s %f\n", str, &opt);
		while(strcmp(str, "COMMENT:") != 0)
			fscanf(f, "%s %f\n", str, &opt);

		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "DIMENSION:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "CAPACITY:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		Q = i;
		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[in1] = in2;
			posy[in1] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		int dmd = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[in1] = in2;
			dmd += in2;
			cnt++;
		}
		k = ceil((float)dmd / Q);
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%f %f\n", &in2, &in3);
		posx[0] = in2;
		posy[0] = in3;
		demands[0] = 0;
		printf("%s\t",argv[1]);

		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "X"))
	{
		ch = getc(f);  while ((ch != EOF) && (ch != 'n')) ch = getc(f);
		ch = getc(f);  i = 0;
		while ((ch != EOF) && (ch != '-'))
		{
			str[i++] = ch; 
			ch = getc(f);
		}
		str[i]='\0';
		n = atoi(str);
		ch = getc(f);  while ((ch != EOF) && (ch != 'k')) ch = getc(f);
		fscanf(f, "%s\n", str);
		k = atoi(str);

		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
		ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

		ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
		fscanf(f, "%s\n", str);
		Q = atoi(str);
		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[cnt] = in2;
			posy[cnt] = in3;
			cnt++;
		}
		fscanf(f, "%s\n", str);
		cnt = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[cnt] = in2;
			cnt++;
		}
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		printf("%s\t",argv[1]);
		opt = 0;
		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}
	else if(strstr(argv[1], "Last"))
	{
		fscanf(f, "%s %f\n", str, &opt);
		while(strcmp(str, "COMMENT:") != 0)
			fscanf(f, "%s %f\n", str, &opt);
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "DIMENSION:") != 0)
			fscanf(f, "%s %d\n", str, &i);
		n = i;
		fscanf(f, "%s %d\n", str, &i);
		while(strcmp(str, "CAPACITY:") != 0)
			fscanf(f, "%s %d\n", str, &i);

		fscanf(f, "%s %f\n", str, &len);
		while(strcmp(str, "DISTANCE:") != 0)
			fscanf(f, "%s %f\n", str, &len);
		Q = i;
		fscanf(f, "%s\n", str);
		while (strcmp(str, "NODE_COORD_SECTION") != 0) 
			fscanf(f, "%s\n", str);

		cnt = 0;
		cudaMallocManaged(&posx, sizeof(float)* n);
		cudaMallocManaged(&posy, sizeof(float)* n);
		cudaMallocManaged(&demands, sizeof(int)* n);

		while (cnt < n) 
		{
			fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
			posx[in1] = in2;
			posy[in1] = in3;
			cnt++;
		}
		fscanf(f, "%d %f %f\n", &in1, &in2, &in3);
		posx[0] = in2;
		posy[0] = in3;
		fscanf(f, "%s\n", str);
		cnt = 0;
		int dmd = 0;
		while (cnt < n) 
		{
			fscanf(f, "%d %f\n", &in1, &in2);
			demands[in1] = in2;
			dmd += in2;
			cnt++;
		}
		k = ceil((float)dmd / Q);
		if (cnt != n) {fprintf(stderr, "read %d instead of %d cities\n", cnt, n);  exit(-1);}
		fscanf(f, "%s", str);
		fscanf(f, "%f %f\n", &in2, &in3);
		demands[0] = 0;
		printf("%s\t",argv[1]);
		create_soln(posx, posy, demands, n, k, Q, f, opt);
	}

	fclose(f);
	cudaFree(posx);
	cudaFree(posy);
	cudaFree(demands);
}
