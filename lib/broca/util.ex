defmodule Broca.Util do
  defmodule RuntimeLogger do
    def log(interval) do
      file = File.open!("./runtime.log", [:write])
      original_leader = Process.group_leader()
      Process.group_leader(self(), file)
      IEx.Helpers.runtime_info()
      Process.group_leader(self(), original_leader)
      File.close(file)

      Process.sleep(interval)

      log(interval)
    end

    # def log_hook(interval) do
    #   log(File.open!("./runtime.log", [:write]), interval)
    # end
    def run(interval \\ 1000) do
      spawn(Broca.Util.RuntimeLogger, :log, [interval])
    end
  end
end
