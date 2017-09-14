# Clothes classification (50 categories) and auto-tagging (1000 attributes) with Keras based on Tensorflow

Run:
```script
python clothes_classifier.py 2>&1 | tee ./result.log
```
The former is a single-label classification problem and the latter is a multi-label classification problem.

- Dataset: [DeepFashion Attribute Prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
  - Number: 289147 images in total
  - Distribution of the Clothes Category:
  
    | category_name | number |
    |---------------|:------:|
    |Anorak         |160     |
    |Blazer         |7495    |
    |Blouse         |24557   |
    |Bomber         |309     |
    |Button-Down    |330     |
    |Cardigan       |13311   |
    |Flannel        |324     |
    |Halter         |17      |
    |Henley         |716     |
    |Hoodie         |4048    |
    |Jacket         |10467   |
    |Jersey         |748     |
    |Parka          |676     |
    |Peacoat        |97      |
    |Poncho         |791     |
    |Sweater        |13123   |
    |Tank           |15429   |
    |Tee            |36887   |
    |Top            |10078   |
    |Turtleneck     |146     |
    |Capris         |77      |
    |Chinos         |527     |
    |Culottes       |486     |
    |Cutoffs        |1669    |
    |Gauchos        |49      |
    |Jeans          |7076    |
    |Jeggings       |594     |
    |Jodhpurs       |45      |
    |Joggers        |4416    |
    |Leggings       |5013    |
    |Sarong         |32      |
    |Shorts         |19666   |
    |Skirt          |14773   |
    |Sweatpants     |3048    |
    |Sweatshorts    |1106    |
    |Trunks         |386     |
    |Caftan         |54      |
    |Cape           |0       |
    |Coat           |2120    |
    |Coverup        |17      |
    |Dress          |72083   |
    |Jumpsuit       |6153    |
    |Kaftan         |126     |
    |Kimono         |2294    |
    |Nightdress     |0       |
    |Onesie         |70      |
    |Robe           |150     |
    |Romper         |7408    |
    |Shirtdress     |0       |
    |Sundress       |0       |

  - Distribution of the Clothes Attributes (Omit)
