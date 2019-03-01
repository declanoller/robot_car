from datetime import datetime
import psutil
import subprocess
import time
import FileSystemTools as fst


class DebugFile:


	def __init__(self, **kwargs):
		#Prepare debug file
		self.date_time_string = fst.getDateString()
		self.debug_fname = 'debug_' + self.date_time_string + '.log'
		self.local_path = kwargs.get('path', './')
		self.full_path_fname = fst.combineDirAndFile(self.local_path, self.debug_fname)
		self.notes = kwargs.get('notes', '')

		self.enabled = kwargs.get('enabled', True)

		self.mqtt_obj = kwargs.get('mqtt_obj', None)

		#self.close_event = file_tool.close_event

		fDebug = open(self.full_path_fname, 'w+')
		fDebug.write('Run notes: ' + self.notes + '\n\n\n')
		fDebug.close()


	def getDateString(self):
		return(self.date_time_string)


	def writeToDebug(self, write_string):

		if self.enabled:

			date_time_string = datetime.now().strftime("[%H:%M:%S") + '.' + str(int(int(datetime.now().strftime("%f"))/1000.0))+datetime.now().strftime("   %Y-%m-%d]")
			whole_string = date_time_string + '\t' + write_string + '\n'

			if self.mqtt_obj is not None:
				self.mqtt_obj.publishDebug(whole_string)

			fDebug = open(self.full_path_fname, 'a+')
			fDebug.write(whole_string)
			fDebug.close()


	def recordTempMemCPU(self):
		CPU_str = 'CPU=' + str(psutil.cpu_percent())
		mem_str = 'Mem=' + str(psutil.virtual_memory()[2])

		temp_cmd = 'cat'
		temp_arg = '/sys/class/thermal/thermal_zone0/temp'
		temp_str = 'temp=' + str(int(subprocess.check_output([temp_cmd,temp_arg]))/1000.0)

		self.writeToDebug(CPU_str + ', ' + mem_str + ', ' + temp_str + '\n')













#
