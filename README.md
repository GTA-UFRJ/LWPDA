# Autor(es): 
- Hugo Antunes
  
# Orientadores: 
- Professor Rodrigo de Souza Couto e
- Pedro Henrique Cruz Caminha

# BetterYolo
Redução do envio de envios de quadros para processamento do YOLO utilizando o OpenCV
Por ora, esse README é apenas um quadro de tarefas a se fazer para melhor organização do trabalho.
Posteriormente, será escrito um resumo, finalidade e manual de como utilizar os códigos dispostos nesse repositório.

# Datas importantes:
- 05/01 > férias do orientador Rodrigo (15 DIAS RESTANTES);
- 15/01 > submissão do artigo (25 DIAS RESTANTES);

# Tarefas a fazer:
- Organização dos códigos (Check);
- Criação da classe para redução de frames; (Check)
- Consolidação de testes utilizando vídeos (ImageNet VID dataset) para o benchmark; (Check)
- Posteriormente, melhorias dos parâmetros utilizados na classe para melhor resultado;
- Leitura da imagem quando quadros iguais (não urgente)
  
# Para o artigo;
- Resumo; (check)
- Abstract; (não urgente)
- Introdução -> Contextualização (Câmeras de segurança), o problema do delay (explicá-lo), Proposta de solução, Comparação de quadros (explicar por alto); (check)
- Desenvolvimento -> Trabalhos relacionados, Metodologia, Experimentos, OpenCV, YOLO, Comparação RGB, Limiares, Dataset, Discussão e resultados etc (Não necessariamente nessa ordem) > *URGENTE*;
- Conclusão -> Trabalhos futuros, Conclusão;
- Realização dos testes com diferentes similaridades e mAP (0 a 100% ~ pular de 10% em 10%); (O código está quase pronto) *URGENTE*;
- Testes comparando os tempos de processamento *URGENTE*;
- Matriz de quadros iguais explicativa -> Cruz; (Desenvolvimento) 
- Gráfico (linear) entre limiar e quadros descartados (60 quadros/segundo ~ 16 ms/quadro);
- Verificar se há um ótimo
