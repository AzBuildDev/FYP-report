"""Offline semantic adapter regression; no weights, microphone, camera or robot."""
import copy
import json
import unittest
from 红蓝语义配置 import *
from 语音任务协议 import compile_request, fixture_scene, DryRunExecutor, InstructionError

class AdapterTests(unittest.TestCase):
    def request(self, color='blue', profile=RED_BLUE_PROFILE, text=None, scene=None, raw=None):
        task = color + '_cube_to_zone_A'
        return compile_request(text or task_texts(RED_BLUE_PROFILE).get(task, 'Put the green cube in zone A.'),
            raw or json.dumps({'status':'parsed','task_id':task}),
            fixture_scene() if scene is None else scene, policy_profile=profile)

    def test_both_training_texts(self):
        for color in ('red', 'blue'):
            with self.subTest(color=color):
                r=self.request(color)
                self.assertEqual(r['status'],'ready')
                self.assertEqual(r['task_text'], task_texts(RED_BLUE_PROFILE)[r['task_id']])
                self.assertFalse(r['executable'])
                self.assertFalse(r['policy_binding']['checkpoint_loaded'])
                self.assertFalse(r['policy_binding']['hardware_validated'])

    def test_legacy_rejects_blue(self):
        self.assertEqual(self.request(profile=LEGACY_PROFILE)['reason'],'policy_not_validated_for_task')

    def test_negation_preserved(self):
        self.assertEqual(self.request(text="Don't take the red cube. Put the blue cube in the target area.")['task_id'],'blue_cube_to_zone_A')

    def test_unsupported_tasks(self):
        for task,text in [('green_cube_to_zone_A','Put the green cube in zone A.'),
                          ('blue_cube_to_zone_B','Put the blue cube in zone B.')]:
            with self.subTest(task=task):
                r=self.request(text=text,raw=json.dumps({'status':'parsed','task_id':task}))
                self.assertEqual(r['reason'],'policy_not_validated_for_task')

    def test_independent_semantic_guard(self):
        for text in ['Put that one over there.', 'Put the red and blue cubes in zone A.', 'Grab the blue cube.']:
            self.assertNotEqual(self.request(text=text)['status'],'ready')
        self.assertEqual(self.request(text=RED_TEXT)['reason'],'model_grounding_mismatch')

    def test_scene_guard(self):
        for change,reason in [({'busy':True},'robot_busy_or_unknown'),
                              ({'observed_at':0},'scene_stale_or_invalid'),
                              ({'objects':[]},'object_not_seen'),
                              ({'zones':[]},'target_not_available')]:
            s=fixture_scene();s.update(change)
            self.assertEqual(self.request(scene=s)['reason'],reason)

    def test_unknown_profile(self):
        self.assertEqual(self.request(profile='invented')['reason'],'unknown_policy_profile')

    def test_profile_and_binding_tamper(self):
        s=fixture_scene();r=self.request(scene=s)
        with self.assertRaises(InstructionError):DryRunExecutor().execute(r,s)
        r['policy_binding']['checkpoint_sha256']='altered'
        with self.assertRaises(InstructionError):DryRunExecutor(RED_BLUE_PROFILE).execute(r,s)

    def test_dry_run_revalidates(self):
        s=fixture_scene();r=self.request(scene=s)
        out=DryRunExecutor(RED_BLUE_PROFILE).execute(r,s)
        self.assertFalse(out['hardware_connected']);self.assertFalse(out['motion_sent'])
        s['observed_at']=0
        with self.assertRaises(InstructionError):DryRunExecutor(RED_BLUE_PROFILE).execute(r,s)

    def test_mapping_isolation(self):
        task_texts(RED_BLUE_PROFILE).clear()
        self.assertEqual(len(task_texts(RED_BLUE_PROFILE)),2)

if __name__=='__main__':unittest.main(verbosity=2)
