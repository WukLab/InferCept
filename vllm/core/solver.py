import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from vllm.sequence import *
import time

class Solver():
  def __init__(self, 
               block_size: int = 16, 
               target: float = 1500, 
               timeout: float = 0.025,
               free_swap_tokens: int = 976,
               per_token_swap_latency: float = 4E-05,
               batch_polynomial: Tuple[float] = (1.3E-05, 0.328, 24.1)) -> None:
    self.block_size = block_size
    self.target = target / self.block_size
    self.timeout = timeout
    self.free_swap = (free_swap_tokens + self.block_size - 1) // self.block_size
    self.per_token_swap_latency = per_token_swap_latency
    self.a, self.b, self.c = batch_polynomial  # ax^2 + bx + c

  def solve(self, 
            sequence_group: SequenceGroup, 
            num_active_gpu_blocks: int, 
            t_call: float,
            t_start: float,
            running_query_head: int,
            running_query_prediction: int,
            swap_in_chunks_head: int,
            swap_in_chunks_tail: int):
    seq: Sequence = sequence_group.get_seqs(status=SequenceStatus.PAUSED_API)[0]
    C: int = seq.data.get_len()
    ret_len: int = sequence_group.sampling_params.api_return_length
    t_arrival: float = sequence_group.arrival_time
    # t_call: float = sequence_group.sampling_params.api_call_time
    api_exec_time: float = sequence_group.sampling_params.api_exec_time

    # get t_resumed from SLA
    t_resumed: float = (C + ret_len) / self.target + t_arrival + api_exec_time
    if t_resumed <= t_start:
      t_resumed = t_start + 0.1     # FIXME use profile time? this is just a max iteration time

    # parse args from tokens to blocks
    C =  (C + self.block_size - 1)  // self.block_size 
    S_gpu = max(num_active_gpu_blocks, C)

    return self._solve(C, 
                       S_gpu, 
                       t_call, 
                       t_start, 
                       t_resumed, 
                       running_query_head,
                       running_query_prediction,
                       swap_in_chunks_head,
                       swap_in_chunks_tail)

  def _solve(self,
            C: int,
            S_gpu: int, 
            t_call: float, 
            t_start: float,
            t_resumed: float,
            running_query_head: int,
            running_query_tail: int,
            c_sin_head: int,
            c_sin_tail: int) -> Tuple[float]:
    # create a model
    m = gp.Model("model")
    m.setParam("NonConvex", 2)
    m.setParam("TimeLimit", self.timeout)
   # m.setParam("LogToConsole", 0)

    # constant
    f_normal = (self.a*(running_query_head)**2 + self.b*(running_query_head) + self.c)/1000

    # decision variables
    c_s = m.addVar(vtype=GRB.INTEGER, name="c_s")
    c_d = m.addVar(vtype=GRB.INTEGER, name="c_d")
    n_e = m.addVar(vtype=GRB.INTEGER, name="n_e")
    b_osout = m.addVar(vtype=GRB.INTEGER, name="b_osout")
    c_osin = m.addVar(vtype=GRB.INTEGER, name="c_osin")
    s_p = m.addVar(vtype=GRB.INTEGER, name="s_p")
    f_swap = m.addVar(vtype=GRB.CONTINUOUS, name="f_swap")
    f_resume = m.addVar(vtype=GRB.CONTINUOUS, name="f_resume")

    # set positivity constraint
    m.addConstr(n_e >= 0)
    m.addConstr(c_s >= 0)
    m.addConstr(b_osout >= 0)
    m.addConstr(c_osin >= 0)
    m.addConstr(c_d >= 0)
    m.addConstr(f_swap >= 0)
    m.addConstr(f_resume >= 0)
    
    # decision constraints
    m.addConstr(n_e * (c_s + c_d) >= 0)
    m.addConstr(n_e * (c_s + c_d) <= C)
    m.addConstr(n_e * (f_resume + self.per_token_swap_latency*(c_osin + c_sin_tail -self.free_swap)*self.block_size) <= (t_resumed - t_start))

    # intermediate constraints to enable max() function and to simplify multilinear expressions
    m.addConstr(s_p == C - n_e * (c_s + c_d))
    m.addConstr(b_osout >= (c_s*n_e+c_sin_head - self.free_swap))  # blocks when swapping out 
    m.addConstr(c_osin >= (c_s+c_sin_tail) - self.free_swap)   # chunks when swapping in

    # functions
    m.addConstr(f_swap == self.per_token_swap_latency*(b_osout + c_sin_head - self.free_swap)*self.block_size)
    m.addConstr(f_resume == (self.a*(running_query_tail+c_d*self.block_size)**2 + self.b*(running_query_tail+c_d*self.block_size) + self.c)/1000)

    # set objective: min (w_p + w_d + w_s + w_o)
    w_p = s_p * (t_resumed - t_call)
    w_d = (n_e + 1) / 2 * c_d * (t_resumed - t_start)
    w_s = (n_e + 1) / 2 * c_s * (t_resumed - t_start) + f_swap * S_gpu
    w_o = S_gpu * (self.per_token_swap_latency*(c_osin + c_sin_tail - self.free_swap)*self.block_size + f_resume - f_normal) * n_e

    m.setObjective(w_p + w_d + w_s + w_o, GRB.MINIMIZE)

    # optimize model
    m.optimize()

    c_s_val = round(m.getVarByName('c_s').X)
    c_d_val = round(m.getVarByName('c_d').X)
    n_e_val = round(m.getVarByName('n_e').X)
    b_osout = round(m.getVarByName('b_osout').X)

    for var in m.getVars():
      print(var.VarName, var.X)
    print(f'f_normal {f_normal}')

    print(f'{C}, {S_gpu}, {t_call}, {t_start}, {t_resumed}, {running_query_head}, {running_query_tail}, {c_sin_head}, {c_sin_tail}, {self.free_swap}')
    #print (c_s_val, c_d_val, n_e_val, b_osout, self.free_swap)
    return  (c_s_val, c_d_val, n_e_val)

def main():
  solver = Solver()

  C = 1024
  api_exec_time = 5
  t_arrival = time.monotonic()
  t_call = t_arrival + 5
  t_start = t_call + api_exec_time
  ret_len = 128
  t_resumed: float = (C + ret_len) / solver.target + t_arrival + api_exec_time
  if t_resumed <= t_start:
    t_resumed = t_start + 0.1     # FIXME use profile time? this is just a max iteration time

  # parse args from tokens to blocks
  C =  (C + solver.block_size - 1)  // solver.block_size 
  num_active_gpu_blocks = 1024
  S_gpu = max(num_active_gpu_blocks, C)

  rqh = 16
  rqt = 16
  csh = (976 + solver.block_size - 1)  // solver.block_size  
  cst = (976 + solver.block_size - 1)  // solver.block_size  

  solver._solve(C,
                S_gpu,
                t_call,
                t_start,
                t_resumed,
                rqh,
                rqt,
                csh,
                cst)

if __name__ == '__main__':
  main()