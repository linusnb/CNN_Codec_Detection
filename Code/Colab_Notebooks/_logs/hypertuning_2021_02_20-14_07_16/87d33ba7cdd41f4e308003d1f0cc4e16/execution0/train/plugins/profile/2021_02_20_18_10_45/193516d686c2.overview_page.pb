?	?i????@?i????@!?i????@	t+?{???t+?{???!t+?{???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?i????@?(?'f@1?????|@A??^
??I Q?@YZ??m??*	??C?<??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??1?M+-@!,?"?X@)X?\T-@1?}??!?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@.=??????!HBɹ? ??).=??????1HBɹ? ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^????!?O?y???)?l˟??1H?e????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchh˹W???!????/\??)h˹W???1????/\??:Preprocessing2F
Iterator::Model@??$"??!???i????)?b?=yx?1?}?Ѥ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9s+?{???Itߞ6?-<@QWO}?Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(?'f@?(?'f@!?(?'f@      ??!       "	?????|@?????|@!?????|@*      ??!       2	??^
????^
??!??^
??:	 Q?@ Q?@! Q?@B      ??!       J	Z??m??Z??m??!Z??m??R      ??!       Z	Z??m??Z??m??!Z??m??b      ??!       JGPUYs+?{???b qtߞ6?-<@yWO}?Q@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI?M~ޝ??!I?M~ޝ??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??dO???!D???C??0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput????Ķ?!??K'????0"=
sequential/conv_layer2/Relu_FusedConv2D??z?薵?!\4?))-??"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???/\??!?h?%?x??0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput????Yg??!??B?$???0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad??EY
|??!???h?F??"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInputX\;?4??!???<.???0"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Bm????!??\]???0"=
sequential/conv_layer3/Relu_FusedConv2D{ ? BO??!??f3Q???Q      Y@YLh/??@@a?Kh/??P@qפ???0@y1BqP??U?"?

both?Your program is POTENTIALLY input-bound because 27.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 