?	Iڍ>>?@Iڍ>>?@!Iڍ>>?@	*?x????*?x????!*?x????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Iڍ>>?@???!9?d@1????~??@A?????	??IĖM??@Y^f?(?@*	??????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?E|'f]@!?`?ф?X@)w0b? 
@1?Bȹ?YX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?U?bٴ?!?t??{??)?U?bٴ?1?t??{??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism	T? ?!??!?q?????)??[[??1Mۦ????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch%?)? ???!??OD????)%?)? ???1??OD????:Preprocessing2F
Iterator::Model'l??ü?!???????)???%z?1??&?J???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9)?x????I??D??"*@QC??r??U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???!9?d@???!9?d@!???!9?d@      ??!       "	????~??@????~??@!????~??@*      ??!       2	?????	???????	??!?????	??:	ĖM??@ĖM??@!ĖM??@B      ??!       J	^f?(?@^f?(?@!^f?(?@R      ??!       Z	^f?(?@^f?(?@!^f?(?@b      ??!       JGPUY)?x????b q??D??"*@yC??r??U@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$?b%???!$?b%???0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW$?????!>???X???0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterMNfg???!d??W????0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputG??Mc??!Vo?z???0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput??1 ??!?B ]???0"=
sequential/conv_layer4/Relu_FusedConv2D????f???!??=iST??"=
sequential/conv_layer3/Relu_FusedConv2DD_I?(y??!?,??????"=
sequential/conv_layer2/Relu_FusedConv2D??	d+???!??­x???"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInputr`i?M??!b????0"\
;gradient_tape/sequential/maxpool_layer3/MaxPool/MaxPoolGradMaxPoolGradP??h???!\XZј???Q      Y@Yf??:D?2@a??L?nPT@q]!u???0@y??)?KE?"?

both?Your program is POTENTIALLY input-bound because 12.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 