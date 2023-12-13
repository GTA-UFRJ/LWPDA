# BetterYolo
Redução do envio de envios de quadros para processamento do YOLO utilizando o OpenCV

Tarefas a fazer:
- Criação da classe para redução de frames; (Check)
- Consolidação de testes utilizando vídeos (ImageNet VID dataset) para o benchmark; (Check)
- Posteriormente, melhorias dos parâmetros utilizados na classe para melhor resultado
- Leitura da imagem quando quadros iguais;
  
Para o artigo (37 Dias restantes):
- Realização dos testes com diferentes thresholds (0 a 100% ~ pular de 10% em 10%);
- Matriz de quadros iguais explicativa;
- Gráfico (linear) entre limiar e quadros descartados (60 quadros/segundo ~ 16 ms/quadro);
- Verificar se há um ótimo
