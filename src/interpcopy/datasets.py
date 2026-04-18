"""Dataset builders for InterpCopy runs.

Single entry point: `murakami_chat_dataset` — loads the COLM paper's
`{instruction, paragraph_text}` JSON and emits samples in chat format so
Llama-3's tokenizer applies its real chat template (BOS, role headers,
eot_id). Matches how Tinker framed the training turns for the COLM paper.
"""

from torchtune.data import InputOutputToMessages
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def murakami_chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "json",
    data_files: str,
    split: str = "train",
    train_on_input: bool = False,
    new_system_prompt: str | None = None,
    **load_dataset_kwargs,
) -> SFTDataset:
    """Build the Murakami-style chat dataset from the COLM paper JSON.

    Each record is mapped to:
      user      -> record["instruction"]
      assistant -> record["paragraph_text"]

    The tokenizer then wraps this in the model's native chat template
    (for llama3_tokenizer: <|begin_of_text|><|start_header_id|>user... etc.).
    """
    message_transform = InputOutputToMessages(
        column_map={"input": "instruction", "output": "paragraph_text"},
        train_on_input=train_on_input,
        new_system_prompt=new_system_prompt,
    )
    return SFTDataset(
        source=source,
        data_files=data_files,
        split=split,
        message_transform=message_transform,
        model_transform=tokenizer,
        **load_dataset_kwargs,
    )
