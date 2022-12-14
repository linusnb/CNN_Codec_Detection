?	???'?|@???'?|@!???'?|@	#?!????#?!????!#?!????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???'?|@((E+?Zd@1???'?r@A??????Iz9??c8
@YSz?????*	??Q?~??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2l?6?#@!\??u%?X@)5bf???"@1??$???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!J???$??);??]??1o??ݠ0??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@75?|έ?!?P7?xK??)75?|έ?1?P7?xK??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch䠄????!&?cjz??)䠄????1&?cjz??:Preprocessing2F
Iterator::ModelY??L/1??!+?@?????)???`?Ht?1*?e?B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9#?!????I?B?OC?A@Q5?\K?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	((E+?Zd@((E+?Zd@!((E+?Zd@      ??!       "	???'?r@???'?r@!???'?r@*      ??!       2	????????????!??????:	z9??c8
@z9??c8
@!z9??c8
@B      ??!       J	Sz?????Sz?????!Sz?????R      ??!       Z	Sz?????Sz?????!Sz?????b      ??!       JGPUY#?!????b q?B?OC?A@y5?\K?O@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????^V??!????^V??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??)??!??;?????0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??!uI??!?ʱ?V???0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?7??"??!?X???e??0"=
sequential/conv_layer2/Relu_FusedConv2DP
?ū?!????\o??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput??7#????!&#T8??0"=
sequential/conv_layer3/Relu_FusedConv2Dfh?
??!??J????"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?(??1A??!
?g????0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad??o-??!@?{?`???"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????!np!???0Q      Y@Y?Z??Z??@aC)?B)Q@q? ŀ'A+@yN#??a?"?

both?Your program is POTENTIALLY input-bound because 35.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 