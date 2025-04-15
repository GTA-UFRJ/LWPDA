# Grupo de Teleinformática e Automação (GTA) - CNPQ/RNP/COPPE/UFRJ


## Autor: 
- Hugo Antunes

## Orientadores: 
- Rodrigo de Souza Couto (rodrigo@gta.ufrj.br)
- Luís Henrique Maciel Kosmalski Costa (luish@gta.ufrj.br)
- Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)

## Artigos publicados
- CoUrb/SBRC disponível em:
- IEEE CloudNet disponível em:


# Light Weight Pixel Difference Accumulation (LWPDA)

## Resumo
A detecção de objetos em tempo real é um desafio comum em diferentes contextos, como em carros autônomos e vigilância por câmeras de segurança. Entretanto, o processamento de vídeos em tempo real exige um alto poder computacional, essa exigência torna comum a ocorrência de atrasos. Essa demora afeta diretamente essas aplicações, já que elas necessitam que a detecção seja feita o quanto antes. Assim, esse repositório propõe uma comparação de quadros sequenciais por meio da utilização dos valores RGB de cada pixel. Aqueles quadros que forem julgados semelhantes não serão enviados para processamento, o que diminui significativamente o tempo de processamento.

## Comparação
Redução do envio de envios de quadros para processamento do YOLO, por meio do descarte de quadros similares. Esse descarte é feito por meio da função "isSimilar" que está encontrada na classe LWPDA em new/LWPDA.py
A similaridade é medida utilizando os valores RGB de cada pixel das duas imagens a serem comparadas.

## Uso
Para a utilização adequada do algoritmo, recomenda-se um limiar entre 40% e 50% do total de valores RGB do vídeo/câmera. Um código exemplo está disponível em codes/UsingMyolo.

