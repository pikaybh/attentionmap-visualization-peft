from utils.attention_visualization import AttentionVisualization, AttentionVisualizationConCat
from typing import Optional, Union
import fire


def main(prompt: Optional[str] = "일용직 작업자가 공사현장에서 비계3층 높이에서 외부비계 해체 작업중 발을 헛디뎌 추락함. 사고당일 병원으로 이송되었으며, 의식불명으로 입원중 사망함", model_id: Optional[str] = "beomi/Llama-3-Open-Ko-8B", peft_model_id: Optional[str] = "./models/Llama-3-Open-Ko-8B-csi-report-acctyp", output: Optional[str] = "i") -> None:
    """Main function to run attention visualization based on the given model and prompt.

    Depending on the output argument, it selects either `AttentionVisualization` or
    `AttentionVisualizationConCat` to handle attention map generation.

    Args:
        prompt (Optional[str]): The input text for the model to generate output. 
            Defaults to a sample accident report in Korean.
        model_id (Optional[str]): The pre-trained model ID to use. Defaults to "beomi/Llama-3-Open-Ko-8B".
        peft_model_id (Optional[str]): The checkpoint of the fine-tuned model. Defaults to "./models/Llama-3-Open-Ko-8B-csi-report-acctyp".
        output (Optional[str]): Determines whether to use `AttentionVisualization` or `AttentionVisualizationConCat`.
            If "i", `AttentionVisualization` is used; otherwise, `AttentionVisualizationConCat` is selected. Defaults to "i".

    Returns:
        None
    """
    attn_vs: Union[AttentionVisualization, AttentionVisualizationConCat] = AttentionVisualization() if output == "i" else AttentionVisualizationConCat()
    attn_vs(prompt, model_id, peft_model_id)


# Main Entry point
if __name__ == "__main__":
    fire.Fire(main)
