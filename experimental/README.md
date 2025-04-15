## 数据

整理到了math_data文件中，分了multimath和mathvista两个文件，multimath有完整的中英文问题、答案、解决过程，mathvista没有那么完整：

``` json
{
        "image_id": "6fb1abf7f9c72c67be68625a0e7d19a0.png",
        "data_type": "geometry",
        "question_type": "填空",
        "level": "2",
        "task_type": "9",
        "QA_pair": [
            {
                "question_zh": "则根据题意可列出方程为________ _.",
                "condition_zh": "如图,在一块长为22米、宽为17米的矩形地面上,要修建同样宽的两条互相垂直的道路(两条道路各与矩形的一条边平行),剩余部分种上草坪,使草坪面积为300平方米.若设道路宽为x米,",
                "answer_zh": "(22-x)(17-x)=300",
                "question_en": "According to the meaning of the question, the equation can be set up as ________ _.",
                "condition_en": "As shown in the figure, on a rectangular ground with a length of 22 meters and a width of 17 meters, two perpendicular roads of the same width are to be built (each road is parallel to one side of the rectangle), and the remaining area is to be covered with lawn, making the lawn area 300 square meters. Let the width of the road be x meters.",
                "solution_zh": "Step 1 (设变量): 设道路的宽为x米。\nStep 2 (确定草坪面积公式): 草坪面积为整个矩形的面积减去道路的面积。\nStep 3 (面积计算): 矩形地面的面积为22米 * 17米 = 374平方米。\nStep 4 (设未知数方程): 根据题意，草坪的面积=（22-x）（17-x）。\nStep 5 (方程等式): (22-x)(17-x)=300。\nAnswer: \\boxed{(22-x)(17-x)=300}",
                "solution_en": "Step 1 (Set the variable): Let the width of the road be x meters.\nStep 2 (Determine the grassland area formula): The area of the grassland is the total area of the rectangle minus the area of the roads.\nStep 3 (Area calculation): The total area of the rectangular ground is 22 m * 17 m = 374 square meters.\nStep 4 (Set the unknown equation): According to the problem, the area of the grassland is (22-x)(17-x).\nStep 5 (Equation formation): (22-x)(17-x)=300.\nAnswer: \\boxed{(22-x)(17-x)=300}"
            }
        ]
    }
```

## 代码
### 评测
评测部分有调用API与开源模型两个文件

- API文件：GPT_API.py

    需要修改API-KEY, 连接网址，输入输出文件路径

    评测时输入对应的语言参数进行语言选择
- 开源模型文件：baseline_test.py 与 eval.sh
    修改模型，输入输出文件路径
    注意：不同的模型可能适配不同的输入prompt格式，现在给出了两种prompt，代码部分可能还有需要完善的。

### 指标
计算指标这里目前还没有完善，现在可以做到的是有标准答案（数据中的answer_zh，与solution末尾\boxed{}中的内容），并提取模型输出\boxed{}中的内容，使用exact_match计算acc，但是现在得到的结果存在一些形式上与标准答案不同，实际结果相同的，这些需要额外进行判断，直接使用exact_match并不准确。

初步的想法，需要进一步完善：

- 编写代码统一格式并直接对比，例如olympaid_bench中的evaluate方法
- 给出参考答案，让LLM进行打分与对比
- 设计更多的metric，可以更多地关注到对符号的理解中
- 人类评估的安排