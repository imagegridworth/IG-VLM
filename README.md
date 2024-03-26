

<h2 align="center"> <a>IG-VLM: An Image Grid Can Be Worth a Video: Zero-shot Video Question Answering Using a VLM</a></h2>
Stimulated by the sophisticated reasoning capabilities of recent Large Language Models(LLMs), a variety of strategies for bridging video modality have been devised. A prominent strategy involves Video Language Models(VideoLMs), which train a learnable interface with video data to connect advanced vision encoders with LLMs. Recently, an alternative strategy has surfaced, employing readily available foundation models, such as VideoLMs and LLMs, across multiple stages for modality bridging. In this study, we introduce a simple yet novel strategy where only a single Vision Language Model (VLM) is utilized. Our starting point is the plain insight that a video comprises a series of images, or frames, interwoven with temporal information. The essence of video comprehension lies in adeptly managing the temporal aspects along with the spatial details of each frame. Initially, we transform a video into a single composite image by arranging multiple frames in a grid layout. The resulting single image is termed as an image grid. This format, while maintaining the appearance of a solitary image, effectively retains temporal information within the grid structure. Therefore, the image grid approach enables direct application of a single high-performance VLM without necessitating any video-data training. Our extensive experimental analysis across ten zero-shot video question answering benchmarks, including five open-ended and five multiple-choice benchmarks, reveals that the proposed Image Grid Vision Language Model (IG-VLM) surpasses the existing methods in nine out of ten benchmarks.

## Requirements and Installation
* Pytorch
* transformers==4.36.2
* Install required packages : pip install -r requirements.txt


## Inference and Evaluation
We provide code that enables the reproduction of our experiments with LLaVA v1.6 7b/13b/34b and GPT-4V using the IG-VLM approach. For each VLM, we offer files that facilitate experimentation across various benchmarks:Open-ended Video Question Answering (VQA) with datasets such as MSVD-QA, MSRVTT-QA, ActivityNet-QA, and TGIF-QA, Text Generation Performance VQA for CI, DO, CU, TU, and CO, Multiple-choice VQA including NExT-QA, STAR, TVQA, IntentQA, and EgoSchema.
 * To conduct these benchmark experiments, please prepare data download and a QA pair sheet. 
 * The QA pair sheet should follow the format outlined below and must be converted into a CSV file for use.
 ```bash
 # for open-ended QA sheet, it should include video_name, question, answer, question_id and question_type(optional)
 # for multiple-choice QA sheet, it should include video_name, question, options(a0, a1, a2, .. ), answer, question_id and question_type(optional).
 # question_id should be unique.

 # example of multeple-choice QA
 | video_name | question_id |                        question                       |       a0      |      a1     |    a2    |        a3      |        a4       |   answer   | question_type(optional) | 
 |------------|-------------|-------------------------------------------------------|---------------|-------------|----------|----------------|-----------------|------------|-------------------------|
 | 5333075105 | unique1234  | what did the man do after he reached the cameraman?   | play with toy |inspect wings|   stop   |move to the side|pick up something|    stop    |            TN           |
 ...
```

 * For experimenting with LLaVA v1.6 combined with IG-VLM, the following command can be used. Please copy the <a href="https://github.com/haotian-liu/LLaVA">LLaVA code</a> to the execution path. The llm_size parameter allows the selection among the 7b, 13b, and 34b model configurations:
 ```bash
 # Open-ended video question answering
 python eval_llava_openended.py --path_qa_pair_csv ./data/open_ended_qa/ActivityNet_QA.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_activitynet/ --api_key {api_key} --llm_size 7b
 ```
 ```bash
 # Text generation performance
  python eval_llava_textgeneration_openended.py --path_qa_pair_csv ./data/text_generation_benchmark/Generic_QA.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_textgeneration/ --api_key {api_key} --llm_size 13b
 ```
 ```bash
 # Multiple-choice VQA
 python eval_llava_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/TVQA.csv --path_video /data/TVQA/videos/%s.mp4 --path_result ./result_tvqa/ --llm_size 34b
 ```
 * When conducting experiments with GPT-4V combined with IG-VLM, the process can be initiated using the following command. Please be aware that utilizing the GPT-4 vision API may incur significant costs. 
 ```bash
 # Open-ended video question answering
 python eval_gpt4v_openended.py --path_qa_pair_csv ./data/open_ended_qa/MSVD_QA.csv --path_video /data/msvd/videos/%s.avi --path_result ./result_activitynet_gpt4/ --api_key {api_key}
 ```
 ```bash
 # Text generation performance
 python eval_gpt4v_textgeneration_openended.py --path_qa_pair_csv ./data/text_generation_benchmark/Generic_QA.csv --path_video /data/activitynet/videos/%s.mp4 --path_result ./result_textgeneration_gpt4/ --api_key {api_key}
 ```
 ```bash
 # Multiple-choice VQA
 python eval_gpt4v_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/EgoSchema.csv --path_video /data/EgoSchema/videos/%s.mp4 --path_result ./result_egoschema_gpt4/ --api_key {api_key}
 ```

