import random

def Q_routing(T,Q,alp,eps,epi_n,start,end):
    num_nodes = [0,0]
    for e in range(epi_n):
        if e in range(0, epi_n,1000):
            print("loop:",e)
        state_cur = start
        #route_cur = [start]
        goal = False

        while not goal:
            mov_valid = list(Q[state_cur].keys())
    
            if len(mov_valid) <= 1:
                state_nxt = mov_valid [0]
            else:
                act_best = random.choice(get_key_of_min_value(Q[state_cur]))
                if random.random() < eps:
                    mov_valid.pop(mov_valid.index(act_best))
                    state_nxt = random.choice(mov_valid)
        
                else:
                    state_nxt = act_best
    
            Q = Q_update(T,Q,state_cur, state_nxt, alp)
    
            if state_nxt in end:
                goal = True

            state_cur = state_nxt

        if e in range(0,1000,200):
            for i in Q.keys():
                for j in Q[i].keys():   
                    Q[i][j] = round(Q[i][j],6)

            nodes = get_bestnodes(Q,start,end)
            num_nodes.append(len(nodes))
            print("nodes:", num_nodes)
            if len(set(num_nodes[-3:])) == 1:
                break

    return Q


def Q_update(T,Q,state_cur, state_nxt, alp):
    t_cur = T[state_cur][state_nxt]
    q_cur = Q[state_cur][state_nxt]
    q_new = q_cur + alp * (t_cur + min(Q[state_nxt].values()) - q_cur)
    
    #q_new = q_cur + alp * (t_cur + gamma * min(Q[state_nxt].values()) )
    #q_cur[action] = reward + self._gamma * np.amax(q_s_a_d[i])
    
    Q[state_cur][state_nxt] = q_new 

    return Q