import argparse
import gc
import io

import torch
from natsort import natsorted, ns

from dataset_configs import RagDataset
from embedding_configs import Embedders, Modality, EncoderConfig, all_embedders


def load_documents(path):
    docs = []
    for child in natsorted(path.iterdir(), alg=ns.IGNORECASE):
        if not child.is_file():
            continue
        file_bytes = child.read_bytes()
        stream = io.BytesIO(file_bytes)
        docs.append(stream)
    return docs

def main(args):
    # args.models：命令行指定的模型名称列表。
    # 例如 ["colnomic", "linq"]。
    #
    # getattr(Embedders, model)：
    # 根据字符串名称，从 Embedders 中取出同名的模型配置。
    #
    # 例如：
    # model = "colnomic"
    # getattr(Embedders, model) 相当于 Embedders.colnomic
    #
    # configs 最终保存所有待运行模型的配置对象。
    # list[EncoderConfig] 是类型提示，表示列表元素应为 EncoderConfig。
    configs: list[EncoderConfig] = [
        getattr(Embedders, model)
        for model in args.models
    ]

    # 进入“不记录梯度”的上下文。
    #
    # 此处只使用已有模型生成嵌入，不需要反向传播，
    # 因此关闭梯度记录，减少相关的显存和计算开销。
    #
    # 注意：torch.no_grad() 不等于 model.eval()，
    # 它本身不会切换 Dropout 等模块的运行模式。
    with torch.no_grad():

        # 外层循环：依次处理每一个模型。
        for config in configs:

            # 根据当前配置，创建对应的嵌入模型封装对象。
            # 模型加载及其处理器等初始化工作，由具体实现负责。
            #
            # 该对象会在当前模型的所有数据集之间复用，
            # 而不是每处理一个数据集就重新创建。
            embedder = config.embedder()

            # 内层循环：用当前模型依次处理指定的数据集。
            for dataset_name in args.datasets:

                # 创建数据集对象。
                #
                # dataset_name：数据集名称或相对路径。
                # prefix：数据集所在位置的路径前缀。
                #
                # 后续通过 dataset.path 定位该数据集的文件。
                dataset = RagDataset(
                    dataset_name,
                    prefix=args.datasets_path_prefix
                )

                # 打印当前使用的模型标识。
                print(f"Running for {config.model_id}")

                # 判断当前模型是否以“页面图像”作为文档输入。
                if config.modality == Modality.VISION:

                    # 找到当前数据集的 images 文件夹。
                    # 这里的 / 是 pathlib 的路径拼接操作。
                    folder = dataset.path / "images"

                    # 获取文件夹中的条目，并进行自然排序。
                    #
                    # 自然排序：
                    # 1.png、2.png、10.png
                    #
                    # 普通字符串排序可能变成：
                    # 1.png、10.png、2.png
                    #
                    # ns.IGNORECASE：排序时忽略大小写。
                    img_docs = natsorted(
                        folder.iterdir(),
                        alg=ns.IGNORECASE
                    )

                    # 只保留文件，排除目录以及 macOS 的 .DS_Store 文件。
                    #
                    # 注意：此时 docs 中存放的是图片文件的路径，
                    # 这里还没有把图片读取成像素或张量。
                    #
                    # 这行代码也没有按图片扩展名进行过滤。
                    docs = [
                        f
                        for f in img_docs
                        if f.is_file() and not f.name == ".DS_Store"
                    ]

                else:

                    # 非视觉分支从 texts 文件夹读取文档文本。
                    # 该分支没有在这里执行 OCR，
                    # 它读取的是已经准备好的文本文件。
                    folder = dataset.path / "texts"

                    # 调用辅助函数加载文档。
                    # 返回的对象支持 read()。
                    text_docs = load_documents(folder)

                    # read()：读取字节内容。
                    # decode()：默认按 UTF-8 解码成 Python 字符串。
                    #
                    # 因此，文本分支中的 docs 是字符串列表。
                    docs = [
                        c.read().decode()
                        for c in text_docs
                    ]

                # 读取该数据集的基准测试表。
                benchmark = dataset.get_benchmark_obj()

                # 取出 question 列中的全部问题，构成查询列表。
                #
                # 这里读取的是问题文本，
                # 没有从 benchmark 中提取相关页面标注作为编码输入。
                queries = [
                    q
                    for q in benchmark["question"]
                ]

                # 核心调用：
                # 将文档和查询交给当前嵌入模型，计算并保存表示。
                #
                # dataset.path：
                #     当前数据集的位置。
                #
                # docs：
                #     视觉分支为图片路径列表；
                #     文本分支为文档字符串列表。
                #
                # queries：
                #     基准中的问题文本列表。
                #
                # config.model_id：
                #     当前模型标识。
                #
                # custom_out_path：
                #     传入自定义输出位置。
                #
                # 图片如何打开、文本如何分词、模型如何前向计算、
                # 向量如何保存，都要继续查看这个方法的具体实现。
                embedder.calc_and_save_embeddings(
                    dataset.path,
                    docs,
                    queries,
                    config.model_id,
                    custom_out_path=args.out_path
                )

                # 主动触发 Python 垃圾回收。
                # 仍被变量引用的对象不会因此被释放。
                gc.collect()

                # 将 PyTorch 缓存中未被使用的显存释放给 CUDA。
                #
                # 它不会释放仍被模型或活跃张量占用的显存。
                torch.cuda.empty_cache()

            # 当前模型处理完所有数据集后，
            # 删除 embedder 对 model 属性的引用。
            #
            # 若模型还被其他对象引用，不能保证立即完全释放。
            del embedder.model

if __name__ == "__main__":
    from dataset_configs import VIDORE1_DATASETS, VIDORE2_DATASETS
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=VIDORE2_DATASETS)
    parser.add_argument('--models', nargs='+', required=True, choices=all_embedders)
    parser.add_argument('--datasets_path_prefix', default='')
    parser.add_argument('--out_path')

    main_args = parser.parse_args()
    main(main_args)
