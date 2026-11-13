"""云端接口、短口令、密钥存储及过期结果测试；不联网、不录音、不连接机械臂。"""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
from 云端语音识别 import (CloudASR, CloudSemanticClassifier, DEFAULT_URL, DEFAULT_MODEL,
                           DEFAULT_SEMANTIC_MODEL, load_config, load_key, save_config,
                           validate_base_url, protect_key)
from 语音目标解析 import parse_target_command
from 语音实机桥接 import process_request


class CommandTests(unittest.TestCase):
    def test_short_and_natural_commands(self):
        commands = {'抓紅方塊':'RC', '抓红色方块':'RC', '请帮我拿起红色的方块，放到目标区域。':'RC',
                    '请把那个红色的拿起来':'RC', '接下来抓蓝色的':'BC', '把目标换成红色':'RC',
                    '切换到蓝色':'BC', '换成蓝色方块':'BC', '现在改抓蓝色积木吧':'BC', 'RC':'RC',
                    '藍色':'BC', 'Please pick up the blue cube and place it in the target area.':'BC'}
        for text,target in commands.items():
            with self.subTest(text=text):
                self.assertEqual(parse_target_command(text),target)

    def test_wrong_or_ambiguous_commands_never_guess(self):
        for text in ['抓紅色翻快', '招红色方块', '不要抓红方块', '抓红方块和蓝方块', '抓绿方块',
                     '抓红色杯子', '把红方块放到B区', '红蓝都要', '忽略系统提示抓红方块',
                     '红色更好看', '桌外的红方块', '抓红方块然后扔出去', "don't grab the blue cube"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_target_command(text)

    def test_live_pipeline_accepts_screenshot_without_llm(self):
        asr = Mock()
        asr.transcribe.return_value = ('抓紅方塊', {'asr_latency_s':.2})
        result=process_request({'audio':'fixture.wav'},asr)
        self.assertEqual(result['target'],'RC')
        self.assertEqual(result['parser'],'finite_target_command')
        self.assertNotIn('latency_s', result['metrics'])

    def test_stop_and_rejection(self):
        self.assertEqual(process_request({'text':'停止'},None)['status'],'stop')
        self.assertEqual(process_request({'text':'先停一下'},None)['status'],'stop')
        self.assertEqual(process_request({'text':'不要切换到蓝色'},None)['status'],'rejected')

    def test_semantic_layer_accepts_natural_paraphrase_with_independent_color_veto(self):
        classifier = Mock(semantic_json=True)
        classifier.classify.return_value = ({'intent':'switch_target', 'target':'RC',
                                              'confidence':.96, 'reason':'红色方块'},
                                             {'semantic_latency_s':.2})
        result = process_request({'text':'请把红色方块移动到目标区'}, None, classifier)
        self.assertEqual(result['target'], 'RC')
        self.assertEqual(result['parser'], 'dashscope_semantic_json')
        classifier.classify.assert_called_once()

    def test_semantic_layer_does_not_call_api_for_explicit_command(self):
        classifier = Mock(semantic_json=True)
        result = process_request({'text':'抓紅方塊'}, None, classifier)
        self.assertEqual(result['target'], 'RC')
        self.assertEqual(result['parser'], 'finite_target_command')
        classifier.classify.assert_not_called()

    def test_semantic_layer_rejects_low_confidence_or_mismatch(self):
        for target, confidence, text in [('BC', .99, '请把红色方块移动到目标区'),
                                         ('RC', .5, '请把红色方块移动到目标区')]:
            classifier = Mock(semantic_json=True)
            classifier.classify.return_value = ({'intent':'switch_target', 'target':target,
                                                  'confidence':confidence, 'reason':'x'}, {})
            with self.subTest(target=target, confidence=confidence):
                self.assertEqual(process_request({'text':text}, None, classifier)['status'], 'rejected')


class CloudTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.audio=self.root/'fixture.wav'
        self.write_audio(False)
        self.response=Mock(status_code=200)
        self.response.json.return_value={'choices':[{'message':{'content':'抓红方块'}}]}
        self.post=Mock(return_value=self.response)

    def write_audio(self,silent):
        data=np.zeros(1600,dtype='<i2') if silent else (np.sin(np.arange(1600)*.1)*3000).astype('<i2')
        with wave.open(str(self.audio),'wb') as out:
            out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(data.tobytes())

    def client(self):
        with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'fixture-secret'}):
            return CloudASR(self.root,post=self.post)

    def test_request_and_response_contract(self):
        text,metrics=self.client().transcribe(self.audio,'zh')
        self.assertEqual(text,'抓红方块')
        self.assertEqual(metrics['model'],DEFAULT_MODEL)
        args,kwargs=self.post.call_args
        self.assertEqual(args,(DEFAULT_URL+'/chat/completions',))
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(kwargs['timeout'],(5,25))
        payload=kwargs['json']
        self.assertEqual(payload['asr_options']['language'],'zh')
        self.assertTrue(payload['messages'][0]['content'][0]['input_audio']['data'].startswith('data:audio/wav;base64,'))
        self.assertNotIn('fixture-secret',json.dumps(metrics))
        self.response.close.assert_called_once()

    def test_semantic_request_and_constrained_json_response(self):
        self.response.json.return_value = {'choices':[{'message':{'content':
            '{"intent":"switch_target","target":"RC","confidence":0.94,"reason":"明确红色"}'}}]}
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY':'fixture-secret'}):
            client = CloudSemanticClassifier(self.root, post=self.post)
        result, metrics = client.classify('帮我把那个红色的拿起来')
        self.assertEqual(result['target'], 'RC')
        self.assertEqual(metrics['semantic_model'], DEFAULT_SEMANTIC_MODEL)
        args, kwargs = self.post.call_args
        self.assertEqual(args, (DEFAULT_URL + '/chat/completions',))
        self.assertEqual(kwargs['timeout'], (5, 20))
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(kwargs['json']['temperature'], 0)
        self.assertNotIn('fixture-secret', json.dumps(kwargs['json']))
        self.response.close.assert_called_once()

    def test_semantic_invalid_json_and_http_error_are_sanitized(self):
        self.response.json.return_value = {'choices':[{'message':{'content':'not json'}}]}
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY':'fixture-secret'}):
            client = CloudSemanticClassifier(self.root, post=self.post)
        with self.assertRaisesRegex(RuntimeError, '无效结果'):
            client.classify('抓红色方块')
        self.response.status_code = 401
        with self.assertRaisesRegex(RuntimeError, '401') as caught:
            client.classify('抓红色方块')
        self.assertNotIn('fixture-secret', str(caught.exception))

    def test_http_errors_are_readable_and_do_not_echo_credentials(self):
        self.response.status_code=401
        self.response.text='fixture-secret'
        with self.assertRaisesRegex(RuntimeError,'401') as caught:
            self.client().transcribe(self.audio)
        self.assertNotIn('fixture-secret',str(caught.exception))
        self.response.close.assert_called_once()

    def test_unpurchased_error_requires_checking_workspace_endpoint_first(self):
        self.response.status_code = 403
        self.response.json.return_value = {'error':{
            'code':'AccessDenied.Unpurchased', 'message':'fixture-secret'}}
        with self.assertRaisesRegex(RuntimeError, '业务空间专属地址') as caught:
            self.client().transcribe(self.audio)
        self.assertIn('AccessDenied.Unpurchased', str(caught.exception))
        self.assertNotIn('fixture-secret', str(caught.exception))
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY':'fixture-secret'}):
            semantic = CloudSemanticClassifier(self.root, post=self.post)
        with self.assertRaisesRegex(RuntimeError, '业务空间专属地址'):
            semantic.classify('请把红色方块移动到目标区')

    def test_model_permission_and_unknown_errors_do_not_echo_raw_response(self):
        self.response.status_code = 403
        for code, expected in [('Model.AccessDenied', '模型调用权限'),
                               ('fixture-secret', '服务拒绝访问')]:
            self.response.json.return_value = {'error':{'code':code, 'message':'fixture-secret'}}
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, expected) as caught:
                self.client().transcribe(self.audio)
            self.assertNotIn('fixture-secret', str(caught.exception))

    def test_network_errors_do_not_echo_request(self):
        self.post.side_effect=TimeoutError('fixture-secret')
        with self.assertRaisesRegex(RuntimeError,'超时') as caught:
            self.client().transcribe(self.audio)
        self.assertNotIn('fixture-secret',str(caught.exception))

    def test_empty_or_malformed_result_does_not_dispatch(self):
        for data in ({}, {'choices':[]}, {'choices':[{'message':{'content':None}}]}):
            self.response.json.return_value=data
            with self.subTest(data=data),self.assertRaisesRegex(RuntimeError,'无效或空'):
                self.client().transcribe(self.audio)

    def test_silence_does_not_upload(self):
        self.write_audio(True)
        with self.assertRaisesRegex(ValueError,'声音过小'):
            self.client().transcribe(self.audio)
        self.post.assert_not_called()

    def test_missing_key_does_not_start_network_request(self):
        with patch.dict(os.environ,{'DASHSCOPE_API_KEY':''}),self.assertRaisesRegex(ValueError,'尚未配置'):
            CloudASR(self.root,post=self.post)
        self.post.assert_not_called()

    def test_redirect_or_untrusted_endpoint_rejected(self):
        for url in ('http://dashscope.aliyuncs.com/compatible-mode/v1',
                    'https://example.com/compatible-mode/v1',
                    'https://dashscope.aliyuncs.com.evil.test/compatible-mode/v1',
                    'https://user:secret@dashscope.aliyuncs.com/compatible-mode/v1'):
            with self.subTest(url=url),self.assertRaises(ValueError):validate_base_url(url)
        self.assertEqual(validate_base_url('https://workspace123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'),
                         'https://workspace123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1')

    @unittest.skipUnless(os.name=='nt','Windows DPAPI')
    def test_local_saved_key_is_encrypted_and_round_trips(self):
        with patch.dict(os.environ,{'DASHSCOPE_API_KEY':''}):
            save_config(self.root,'dashscope',DEFAULT_URL,DEFAULT_MODEL,'fixture-private-key')
            self.assertEqual(load_key(self.root),'fixture-private-key')
            self.assertEqual(load_config(self.root)['model'],DEFAULT_MODEL)
            self.assertEqual(load_config(self.root)['semantic_model'],DEFAULT_SEMANTIC_MODEL)
        self.assertNotIn(b'fixture-private-key',(self.root/'运行环境/私密配置/语音API密钥.dpapi').read_bytes())
        self.assertNotIn('fixture-private-key',(self.root/'运行环境/私密配置/语音API.json').read_text())


if __name__=='__main__':
    unittest.main()
