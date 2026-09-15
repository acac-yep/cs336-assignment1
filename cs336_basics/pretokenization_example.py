import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立统计的多个数据块。
    如果边界重叠，最终返回的块数可能少于目标数量。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件的总字节数。
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 先生成均匀分布的边界初始位置。
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向后读取 4 KB。

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从边界初始位置开始查找。
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小段数据。

            # 如果到达文件末尾，就把文件末尾作为边界。
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 将边界移动到下一个特殊 token 的位置。
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去除重复边界，因此最终块数可能少于目标块数。
    return sorted(set(chunk_boundaries))


## 使用示例
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 每个 [start, end) 区间都可以独立处理。
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对当前数据块做预分词，并统计每种预分词的次数。
