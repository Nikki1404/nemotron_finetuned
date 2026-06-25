(venv) PS C:\Users\re_nikitav\Documents\nemotron_asr> python .\client.py --file C:\Users\re_nikitav\Downloads\audios\audios\audio_maria\maria7.mp3 --language en-US --realtime
[warn] Health check failed: HTTP Error 404: Not Found  (server may still be starting)
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\nemotron_asr\client.py", line 322, in <module>
    main()
    ~~~~^^
  File "C:\Users\re_nikitav\Documents\nemotron_asr\client.py", line 311, in main
    asyncio.run(
    ~~~~~~~~~~~^
        run_file(
        ^^^^^^^^^
    ...<4 lines>...
        )
        ^
    )
    ^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Documents\nemotron_asr\client.py", line 160, in run_file
    with wave.open(str(wav_path), "rb") as wf:
         ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\wave.py", line 659, in open
    return Wave_read(f)
  File "C:\Program Files\Python313\Lib\wave.py", line 286, in __init__
    self.initfp(f)
    ~~~~~~~~~~~^^^
  File "C:\Program Files\Python313\Lib\wave.py", line 253, in initfp
    raise Error('file does not start with RIFF id')
wave.Error: file does not start with RIFF id
