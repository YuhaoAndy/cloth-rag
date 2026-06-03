import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id  # 会话id
        self.storage_path = storage_path  # 不同会话id的存储文件，所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._ensure_history_file()

    def _ensure_history_file(self) -> None:
        """确保历史文件存在且是合法 JSON 数组。"""
        if not os.path.exists(self.file_path):
            self._write_json([])
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                self._write_json([])
                return
            data = json.loads(content)
            if not isinstance(data, list):
                self._write_json([])
        except (json.JSONDecodeError, OSError, ValueError):
            self._write_json([])

    def _write_json(self, data) -> None:
        """原子写入，避免中断造成空文件或坏 JSON。"""
        tmp_path = self.file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, self.file_path)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence序列 类似list、tuple
        all_messages = list(self.messages)  # 已有的消息列表
        all_messages.extend(messages)  # 新的和已有的融合成一个list

        # 将数据同步写入到本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        # 官方message_to_dict：单个消息对象（BaseMessage类实例） -> 字典
        # new_messages = []
        # for message in all_messages:
        #     d = message_to_dict(message)
        #     new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]
        self._write_json(new_messages)

    @property  # @property装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                messages_data = json.loads(content)
                if not isinstance(messages_data, list):
                    return []
                return messages_from_dict(messages_data)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return []

    def clear(self) -> None:
        self._write_json([])
