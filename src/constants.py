RANDOM_SEED = 993

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

TEXT_COLUMNS = [
    "query",
    "product_title",
    "product_description",
    "product_bullet_point",
]

CATEGORICAL_COLUMNS = ["product_brand", "product_color"]

DROP_COLUMNS = ["product_locale"]

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

EMBEDDING_BATCH_SIZE = 16
