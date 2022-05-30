import math
import dists

# goal sempre sera 'bucharest'
def a_star(start, goal='Bucharest'):

    """
    Retorna uma lista com o caminho de start até 
    goal segundo o algoritmo A*
    """
    
    """Dados Iniciais"""
    no_atual = start
    nos_avaliados = []
    
    """ Estrutura que guarda as bordas 
    {'Cidade':{'CustoOrigem':'Custo caminho origem','CustoSolucao':'Custo estimado solucao'}"""
    bordas = {}
    
    """Incluir nó inicial"""
    bordas[start] = {'CustoOrigem':0,
                     'CustoSolucao':dists.straight_line_dists_from_bucharest[start]}
    nos_avaliados.append(start)
    
    
    
    while no_atual != goal:
        
        """Pega a distancia percorrida até agora"""
        distancia_percorrida = bordas[no_atual]['CustoOrigem']
    
        """Tira o nó atual da lista de bordas"""
        bordas.pop(no_atual)
      
        """Pega todos os nós vinculados ao nó atual"""
        nos_filhos = dists.dists[no_atual]
        
        for no in nos_filhos:
            """Insere as novas bordas na lista"""            
            bordas[no[0]] = {'CustoOrigem':distancia_percorrida + no[1],
                            'CustoSolucao':distancia_percorrida + no[1]
                            +dists.straight_line_dists_from_bucharest[no[0]]}
                              
        
        melhor_custo = 0
        """Avalia todas as bordas e seleciona a de melhor custo"""
        for cidade,custos in bordas.items():

            if (melhor_custo == 0 or (melhor_custo > custos['CustoSolucao'])):
                melhor_custo = custos['CustoSolucao']
                no_atual = cidade
        
        """Insere o nó escolhido na lista de avaliados"""
        if (no_atual not in nos_avaliados):    
            nos_avaliados.append(no_atual)        
        
           
    print("Nós avaliados:")
    print(nos_avaliados)
    
    print("Melhor Caminho:")
    index = -1
    """ Avalia o melhor caminho navegando na lista do destino até a origem,
        Retirando os nós que não estão vinculados"""
    while (nos_avaliados[index] != start):
        
        indexPai = index - 1
        isPai = False
        
        while not isPai:
            for cidade in dists.dists[nos_avaliados[index]]:
                if cidade[0] == nos_avaliados[indexPai]:
                    isPai = True
                    break
            if not isPai:
                nos_avaliados.remove(nos_avaliados[indexPai])
        
        index = index - 1
    print(nos_avaliados)
    
a_star('Zerind')