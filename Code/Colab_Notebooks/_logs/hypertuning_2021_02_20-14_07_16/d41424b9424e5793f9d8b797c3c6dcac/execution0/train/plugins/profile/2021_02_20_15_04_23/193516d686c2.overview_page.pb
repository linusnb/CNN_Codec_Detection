?	?#0`c?@?#0`c?@!?#0`c?@	'?ǯ?k?'?ǯ?k?!'?ǯ?k?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?#0`c?@??????c@1&VF#?Е@A???t ???IP?R)?@Y9?⪲???*	?V??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????#@!}/{??X@)?????"@1rq???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@{???`Ĳ?!>?Ȍ???){???`Ĳ?1>?Ȍ???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch%?}?e???!??????)%?}?e???1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????%??!?zc?+???)i???>Ȓ?1~?m?????:Preprocessing2F
Iterator::Model!??i??!AhB?q??)J??{dsu?1tL???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'?ǯ?k?I??x}?%@Q?]?V?\V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????c@??????c@!??????c@      ??!       "	&VF#?Е@&VF#?Е@!&VF#?Е@*      ??!       2	???t ??????t ???!???t ???:	P?R)?@P?R)?@!P?R)?@B      ??!       J	9?⪲???9?⪲???!9?⪲???R      ??!       Z	9?⪲???9?⪲???!9?⪲???b      ??!       JGPUY'?ǯ?k?b q??x}?%@y?]?V?\V@?"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterA[o?c???!A[o?c???0"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter	???b|??!%?Cc??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???B???!????U??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput????ǰ?!?^??C??0"=
sequential/conv_layer2/Relu_FusedConv2DxI??????!??t??V??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?}? 4 ??!?????6??0"=
sequential/conv_layer3/Relu_FusedConv2D???-N??!????n???"=
sequential/conv_layer4/Relu_FusedConv2DT?}?B??!Q?(???"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInputu#???!?2(????0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad{???^??!???~{??Q      Y@Y?ڴK??2@aS?mKCT@qT?ٿv?7@y??97???"?

both?Your program is POTENTIALLY input-bound because 10.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 