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
	int gap1, gap2, j, curr, curr2;
	int anc1, anc2, suc1, suc2;
	
	long change = 0;
	float change1, change2;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (rt_nodes[i] >= 1 && i < k)
	{
		curr = rt_st[i];
		anc1 = 0;
		if(k_routes[curr] != 0)
			suc1 = k_routes[curr];
		else
			suc1 = 0;
		while(curr != 0)
		{
			gap1 = demands[curr] + Q - cap[i];
			for(j = i + 1; j < k; j++ )
			{
				if(rt_nodes[j] > 1)
				{
					curr2 = rt_st[j];
					anc2 = 0;
					suc2 = k_routes[curr2];
					while( curr2 != 0)
					{
						gap2 = demands[curr2] + Q - cap[j];
						if(demands[curr2] <= gap1 && demands[curr] <= gap2)
						{
							change1 = distD(anc1, curr2, posx, posy)
								+ distD(curr2, suc1, posx, posy)
								- ( distD(anc1, curr, posx, posy) 
								+ distD(curr, suc1, posx, posy));

							change2 = distD(anc2, curr, posx, posy)
								+ distD(curr, suc2, posx, posy)
								- (distD(anc2, curr2, posx, posy) 
								+ distD(curr2, suc2, posx, posy));
							change = (change1 + change2) * 100; 
							if(change < sd[i].tot_change)
							{
								sd[i].change_1 = change1;
								sd[i].change_2 = change2;
								sd[i].tot_change = change;
								sd[i].rt_1 = i;
								sd[i].rt_2 = j;
								sd[i].ct_1 = curr;
								sd[i].ct_2 = curr2;					
							}
						}
						anc2 = curr2;
						curr2 = suc2;
						if(curr2 != 0)
							if (k_routes[curr2] != 0)
								suc2 = k_routes[curr2];
							else
								suc2 = 0;
					}
				}
			}  
			anc1 = curr;
			curr = k_routes[curr];
			if(curr != 0)
			{
				if (k_routes[curr] != 0)
					suc1 = k_routes[curr];
				else
					suc1 = 0;
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
	if(k < 128)
	{
		blk = 1;
		thrd = k;
	}
	else
	{
		thrd = 128;
		blk = (k - 1)/ thrd + 1;
	}

	struct swap_data * sd;
	cudaMallocManaged(&sd, sizeof(struct swap_data )* k);
	init_swap<<<blk, thrd>>>(sd, k);
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
			init_swap<<<blk, thrd>>>(sd, k);
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
struct relocate_data
{
	int rt_1, rt_2, ct_1, ct_2;
	float change_1, change_2;
	long tot_change;
};

/* A function calculates an effect of removing a customer from one route */
__device__ float rm_diff(int *k_routes, int st, int j, float *posx, float *posy)
{
	float change2; int st1;
	if(st == j)
	{
		if(k_routes[st] != 0)
		{
			change2 = distD(0, k_routes[st], posx, posy) 
				- distD(0, j, posx, posy) 
				- distD(k_routes[st], j, posx, posy);
		}
		else
			change2 = - (2 * distD(0, j, posx, posy));
	}
	else
	{
		st1 = k_routes[st];
		while(st1 != j)
		{
			st1 = k_routes[st1];
			st = k_routes[st];
		}
		if(st1 != 0)
		{
			change2 = distD(st, k_routes[st1], posx, posy)
				- distD(st, j, posx, posy) 
				- distD(k_routes[st1], j, posx, posy);
		}
		else
		{
				change2 = - ( distD(st, j, posx, posy) 
					+ distD(0, j, posx, posy) 
					- distD(0, st, posx, posy)); 
		}
	}
return change2;
}
/* relocate heuristic is applied after feasible solution is constructed */
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
/* A GPU kernel for relocate heuristic */
__global__ void relocate_kernel(int *k_routes, int *rt_st, float*posx, float*posy, int * demands, int * cap, int Q, int n, int *rt_nodes, struct relocate_data * sd, int *rt_id, int k)
{
	int avl_cap, j;
	int *nbr;
	nbr = (int *)malloc(sizeof(int));
	long change;
	float change1, change2;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < k)
	{	
		avl_cap = Q - cap[i];
		if(rt_nodes[i] >= 1 && avl_cap)
		for(j = i+1; j < k; j++)
		{	
			if(rt_nodes[j] > 1 )
			{
				int j_ct = rt_st[j];
				while(j_ct != 0)
				{
					if(demands[j_ct] <= avl_cap)
					{
						change1 = add_diff(k_routes, rt_st, j_ct, posx, posy, rt_nodes[i], nbr, i);
						change2 = rm_diff(k_routes, rt_st[j], j_ct, posx, posy);
						change = (change1 + change2) * 100; 
						if(change < sd[i].tot_change)
						{
							sd[i].rt_1 = i;
							sd[i].rt_2 = j;
							sd[i].ct_1 = j_ct;
							sd[i].ct_2 = *nbr;
							sd[i].change_1 = change1;
							sd[i].change_2 = change2;
							sd[i].tot_change = change;
						}
					}
					j_ct = k_routes[j_ct];
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
/* this function is made to distribute work of relocate heuristic over GPU  */
float relocate(int *k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int *demands, int Q, int *cap, int*rt_id, int n)
{
	int flag = 1;
	int thrd, blk;
	if(k < 128)
	{
		blk = 1;
		thrd = k;
	}
	else
	{
		thrd = 128;
		blk = (k - 1)/ thrd + 1;
	}
	struct relocate_data * sd;
	cudaMallocManaged(&sd, sizeof(struct relocate_data )* k);
	init_relocate<<<blk, thrd>>>(sd, k);
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

		find_min_relocate(sd, k);
		if(sd[0].tot_change < 0)
		{
			itr_rm_cust(k_routes, rt_st, rt_st[sd[0].rt_2], sd[0].rt_2, sd[0].ct_1);
			itr_add_cust(k_routes, rt_st, rt_st[sd[0].rt_1], sd[0].ct_1, sd[0].rt_1, sd[0].ct_2);
				
			cap[sd[0].rt_2] = cap[sd[0].rt_2] - demands[sd[0].ct_1];
			cap[sd[0].rt_1] = cap[sd[0].rt_1] + demands[sd[0].ct_1];
			rt_dst[sd[0].rt_1] +=  sd[0].change_1;
			rt_dst[sd[0].rt_2] +=  sd[0].change_2;
			rt_nodes[sd[0].rt_2]--;
			rt_id[sd[0].ct_1] = sd[0].rt_1;
			dst += ((float)sd[0].tot_change/100);
			rt_nodes[sd[0].rt_1]++;
			if(rt_st[sd[0].rt_2] == 0)
			{
				rt_nodes[sd[0].rt_2] = 0;
				cap[sd[0].rt_2] = 0;
				rt_dst[sd[0].rt_2] = 0;
			}
			flag = 1;
			init_relocate<<<blk, thrd>>>(sd, k);
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
/* A kernel function to perform 3-opt on GPU */
__global__ void three_opt_kernel(int*k_routes, int*rt_nodes, int*rt_st, float *rt_dst, float *posx, float *posy, int k)
{
int j, i, st, x, y, z, m;
float change, flag;
int curr1, curr2, curr3;
int prev, i_next, j_next, m_next = 0, next; 
long lchange, minchange = 0;
st = threadIdx.x + blockIdx.x * blockDim.x;

if(st < k)
{
	if(rt_nodes[st] > 3 )
	{
		flag = 1;
		while(flag)
		{
			flag = 0; curr1 = rt_st[st];
			for(i = 0; i < (rt_nodes[st] - 2); i++)
			{	
				curr2 = k_routes[curr1];
				for(j = i + 1; j < (rt_nodes[st] - 1); j++)
				{
					curr3 = k_routes[curr2];
					for(m = j + 1; m < rt_nodes[st]; m++)
					{
						m_next = k_routes[curr3];
						change = distD(curr1, curr3, posx, posy) 
						+ distD(k_routes[curr1], k_routes[curr2], posx, posy) 
						+ distD(curr2, m_next, posx, posy) 
						- ( distD(curr1, k_routes[curr1], posx, posy)
						+ distD(curr2, k_routes[curr2], posx, posy)
						+ distD(curr3, m_next, posx, posy));
						lchange = change * 100;	
						if(lchange < minchange)
						{
							x = curr1;
							y = curr2;
							z = curr3;
							minchange = lchange;
						}

						curr3 = k_routes[curr3];
					}
					curr2 = k_routes[curr2];
				}
				curr1 = k_routes[curr1];
			}
			if(minchange < 0)
			{
				j = rt_st[st];		
				flag = 1;
				i_next = k_routes[x];
				j_next = k_routes[y];
				m_next = k_routes[z];
				if(j_next != z)
				{
					next = k_routes[j_next];
					k_routes[j_next] = i_next;
					prev = j_next;
					while(next != z)
					{
						i = k_routes[next];
						k_routes[next] = prev;
						prev = next;
						next = i;
					}
					k_routes[z] = prev;
					k_routes[x] = z;
					k_routes[y] = m_next;
				}
				else
				{
					k_routes[j_next] = i_next;
					k_routes[x] = z;
					k_routes[y] = m_next;
				}
				rt_dst[st] += (float)minchange/100;
				minchange = 0;			
			}
		}
	}
}
}
/* This function is made to distribute 3-opt heuristic work over GPU */
float three_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n)
{
	int thrd, blk;
	if(k < 128)
	{
		blk = 1;
		thrd = k;
	}
	else
	{
		thrd = 128;
		blk = (k - 1) / thrd + 1;
	}
	int *d_k_routes, *d_rt_st, *d_rt_nodes;
	float *d_rt_dst;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_dst, sizeof(float) * k);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_dst, rt_dst, sizeof(float)* k, cudaMemcpyHostToDevice);

	three_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_nodes, d_rt_st, d_rt_dst, posx, posy, k);

	cudaMemcpy(k_routes, d_k_routes, sizeof(int)* n, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_st, d_rt_st, sizeof(int)* k, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_dst, d_rt_dst, sizeof(float)* k, cudaMemcpyDeviceToHost);
	dst = 0;
	for(int i = 0; i < k; i++)
		dst +=rt_dst[i];

	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_rt_dst);

	return dst; 
}
/* A kernel function is made to perform 2-opt heuristic work over GPU */
__global__ void two_opt_kernel(int *k_routes, int *rt_nodes, int *rt_st, float *rt_dst, float *posx, float *posy, int k)
{
int j, i, st, x, y, flag;
long change, minchange =0;
int curr, curr2, i_next = 0, j_next, next, prev, tmp;
st = threadIdx.x + blockDim.x * blockIdx.x;

if(st < k)
{
	if(rt_nodes[st] > 2 )
	{
		flag = 1;
		minchange = 0;
		while(flag)
		{
			i = 0;	curr = rt_st[st];
			flag = 0;
			while(curr != 0)
			{	
				j = curr;
				curr2 = k_routes[curr];
				while(j != 0)
				{
					change = 100*( distD(i, j, posx, posy) 
					+ distD(curr, curr2, posx, posy) 
					- ( distD(i, curr, posx, posy)
					+ distD(j, curr2, posx, posy))  );
					
					if(change < minchange)
					{
						x = i;
						y = j;
						minchange = change;
					}
					j = curr2;
					curr2 = k_routes[curr2];
				}
				i = curr;
				curr = k_routes[curr];
			}
			if(minchange < 0)
			{

				flag = 1;
				rt_dst[st] += (float)minchange/100;

				j_next = k_routes[y];
				if(x == 0)
				{
					i_next = rt_st[st];
					rt_st[st] = y;
				}
				else
				{
					i_next = k_routes[x];
					k_routes[x] = y;
				}
				next = k_routes[i_next];
				k_routes[i_next] = j_next;
				prev = i_next;
				while(next != y)
				{
					tmp = k_routes[next];
					k_routes[next] = prev;
					prev = next;
					next = tmp;
				}
				k_routes[y] = prev;
				minchange = 0;
			}
		}
	}
}
}
/* This function distribute 2-opt heuristic work over GPU */
float two_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n)
{
	int thrd, blk;
	if(k < 128)
	{
		blk = 1;
		thrd = k;
	}
	else
	{
		thrd = 128;
		blk = (k - 1) / thrd + 1;
	}
	int *d_k_routes, *d_rt_st, *d_rt_nodes;
	float *d_rt_dst;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_dst, sizeof(float) * k);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_dst, rt_dst, sizeof(float)* k, cudaMemcpyHostToDevice);

	two_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_nodes, d_rt_st, d_rt_dst, posx, posy, k);

	cudaMemcpy(k_routes, d_k_routes, sizeof(int)* n, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_st, d_rt_st, sizeof(int)* k, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_dst, d_rt_dst, sizeof(float)* k, cudaMemcpyDeviceToHost);
	dst = 0;
	for(int i = 0; i < k; i++)
		dst +=rt_dst[i];

	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	return dst; 
}
/* A kernel function is made for or-opt to work over GPU  */
__global__ void or_opt_kernel(int *k_routes, int *rt_nodes, int *rt_st, float *rt_dst, float *posx, float *posy, int k)
{
int j, i, x, y, stride, first_prev, first, last, last_next;
int min_stride, min_i, min_j, j_next;
float change;
int flag, count1, count2;
int curr, curr2, i_prev = 0, j_prev = 0, next; 
long lchange, minchange = 0;
x = threadIdx.x + blockIdx.x * blockIdx.x;

if(x < k)
{
	if(rt_nodes[x] > 2 )
	{
		flag = 1;
		while(flag)
		{
			flag = 0;
			if(rt_nodes[x] == 3)
				stride = 2;
			else
				stride = 3;
			for(; stride > 0; stride--)
			{
				count1 = 0;i_prev = 0;curr = rt_st[x];
				while(count1 < (rt_nodes[x] - 1 - stride))
				{
					first_prev = i_prev;
					first = curr; j = curr; 
					for(count2 = 0; count2 < (stride - 1); count2++)
						j = k_routes[j];
					last = j;
					last_next = k_routes[last];
					curr2 = last_next;
					for(count2 = count1 + stride; count2 < (rt_nodes[x] - 1); count2++)
					{


						j_next = k_routes[curr2];
						change = distD(curr2, first, posx, posy)
							+ distD(j_next, last, posx, posy)
							+ distD(first_prev, last_next, posx, posy)
							- ( distD(first_prev, first, posx, posy)
							+ distD(last, last_next, posx, posy)
							+ distD(j_next, curr2, posx, posy));
						lchange = change * 100;
						if(lchange < minchange)
						{
							minchange = lchange;
							min_i = curr;
							min_j = curr2;
							min_stride = stride;
						}
						curr2 = j_next;
					}
					i_prev = curr;
					curr = k_routes[curr];
					count1++;
				}
			}
			if(minchange < 0)
			{
				flag = 1;
				rt_dst[x] += (float)minchange/100;
				j = rt_st[x];		
				i = rt_st[x];
				while(i != min_i)
				{
					i_prev = i;
					i = k_routes[i];
				}
				j = 0; y = i;
				while( j < min_stride)
				{
					j_prev = y;
					y = k_routes[y];
					j++;
				}
				if(min_i == rt_st[x])
				{
					rt_st[x] = y;
					while(y != min_j)
						y = k_routes[y];
					next = k_routes[y];
					k_routes[y] = i;
					k_routes[j_prev] = next;
				}
				else
				{
					k_routes[i_prev] = y;
					while(y != min_j)
						y = k_routes[y];
					next = k_routes[y];
					k_routes[y] = i;
					k_routes[j_prev] = next;
				}
				j = rt_st[x];		
				minchange = 0;
			}
		}
	}
}
}
/* This function distribute work of or-opt over GPU */
float or_opt(int*k_routes, int *rt_st, int *rt_nodes, float *rt_dst, float *posx, float *posy, int k, float dst, int n)
{
	int thrd, blk;
	if(k < 128)
	{
		blk = 1;
		thrd = k;
	}
	else
	{
		thrd = 128;
		blk = (k - 1) / thrd + 1;
	}
	int *d_k_routes, *d_rt_st, *d_rt_nodes;
	float *d_rt_dst;
	cudaMalloc(&d_k_routes, sizeof(int) * n);
	cudaMalloc(&d_rt_st, sizeof(int) * k);
	cudaMalloc(&d_rt_nodes, sizeof(int) * k);
	cudaMalloc(&d_rt_dst, sizeof(float) * k);

	cudaMemcpy(d_k_routes, k_routes, sizeof(int)* n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_st, rt_st, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_nodes, rt_nodes, sizeof(int)* k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rt_dst, rt_dst, sizeof(float)* k, cudaMemcpyHostToDevice);

	or_opt_kernel<<<blk, thrd>>>(d_k_routes, d_rt_nodes, d_rt_st, d_rt_dst, posx, posy, k);

	cudaMemcpy(k_routes, d_k_routes, sizeof(int)* n, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_st, d_rt_st, sizeof(int)* k, cudaMemcpyDeviceToHost);
	cudaMemcpy(rt_dst, d_rt_dst, sizeof(float)* k, cudaMemcpyDeviceToHost);
	dst = 0;
	for(int i = 0; i < k; i++)
		dst +=rt_dst[i];

	cudaFree(d_k_routes);
	cudaFree(d_rt_st);
	cudaFree(d_rt_nodes);
	cudaFree(d_rt_dst);

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
		d1 = relocate(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, demands, Q, cap, rt_id, n);
		if (d1 < dst)
		{
			d1 = two_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n);
			d1 = or_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n);
			d1 = three_opt(k_routes, rt_st, rt_nodes, rt_dst, posx, posy, k, d1, n);
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
	cudaSetDevice(1);
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
/*		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
/*		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
//		printf("\nIt is in right place...\n");
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
//		printf("\nInstance has been captured successfully...!!!\n");
		printf("%s\t",argv[1]);
		create_soln(posx, posy, demands, n, k, Q, f, 0);
//		create_soln_2(posx, posy, demands, n, k, Q, f, 0);
//		for(i = 0; i < n; i++)
//			printf("%d. %f\t%f\t%d\n", i, posx[i], posy[i], demands[i]);
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
/*		posx = (float *)malloc(sizeof(float) * n);  
		if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
		posy = (float *)malloc(sizeof(float) * n);  
		if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
		demands = (int *)malloc(sizeof(int) * n);  
		if (demands == NULL) {fprintf(stderr, "cannot allocate demands\n");  exit(-1);}	
*/
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
